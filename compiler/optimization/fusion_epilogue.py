import statistics
import time

import torch
import torch.fx as fx
import triton
from torch._dispatch.python import no_python_dispatcher
from torch.utils._python_dispatch import _disable_current_modes

from .fusion_pointwise import (
    _tensor_meta,
    build_fused_op,
    emit_pointwise_ops,
    external_inputs,
    find_fusion_groups,
    load_kernel,
    new_kernel_name,
)

aten = torch.ops.aten

MATMUL_OPS = (aten.addmm.default, aten.mm.default)
DOT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

TL_DTYPE = {
    torch.float16: "tl.float16",
    torch.bfloat16: "tl.bfloat16",
    torch.float32: "tl.float32",
}

# Candidate tile shapes; the autotuner picks the fastest per (M, N, K).
AUTOTUNE_CONFIGS = """[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=4),
    ]"""


def _is_static_cuda(val):
    return val.is_cuda and all(isinstance(s, int) for s in val.shape)


def find_epilogue_anchor(group: list[fx.Node], inputs: list[fx.Node]):
    """Return the matmul node whose output this group can consume straight
    from the accumulator, or None.

    The matmul must be 2D, every one of its users must be inside the group
    (otherwise its raw output still has to be stored), and the group's output
    must have the matmul's shape so each program can finish its own tile.
    """
    members = set(group)
    out_shape = _tensor_meta(group[-1]).shape
    for node in inputs:
        if node.op != "call_function" or node.target not in MATMUL_OPS or node.kwargs:
            continue  # kwargs means beta/alpha scaling; not handled
        mat1, mat2 = node.args[-2:]
        val, a, b = _tensor_meta(node), _tensor_meta(mat1), _tensor_meta(mat2)
        if (
            all(u in members for u in node.users)
            and val.dim() == 2
            and val.shape == out_shape
            and all(_is_static_cuda(t) for t in (val, a, b))
            and a.dtype == b.dtype
            and a.dtype in DOT_DTYPES
        ):
            return node
    return None


def generate_kernel_source(name, group, matmul, inputs):
    """Tiled GEMM (grouped program order for L2 reuse) whose epilogue runs
    the bias add and the pointwise group on the accumulator in registers.

    `inputs` are the extra tensors the epilogue reads (bias, residual,
    scale, ...). Each is indexed with its own (m, n) strides, so broadcast
    dims have stride 0 and no copies are made.
    """
    in_dtype = _tensor_meta(matmul.args[-2]).dtype
    out_dtype = _tensor_meta(group[-1]).dtype
    # Match eager: fp32 matmuls only use TF32 when PyTorch is told to.
    if in_dtype == torch.float32 and not torch.backends.cuda.matmul.allow_tf32:
        dot_args = ', input_precision="ieee"'
    else:
        dot_args = ""

    in_ptrs = [f"in{i}_ptr" for i in range(len(inputs))]
    in_strides = [f"in{i}_s{d}" for i in range(len(inputs)) for d in "mn"]
    params = ",\n    ".join(
        ["a_ptr", "b_ptr", "out_ptr"] + in_ptrs + ["M", "N", "K"]
        + ["stride_am", "stride_ak", "stride_bk", "stride_bn", "stride_om", "stride_on"]
        + in_strides
        + ["BLOCK_M: tl.constexpr", "BLOCK_N: tl.constexpr",
           "BLOCK_K: tl.constexpr", "GROUP_SIZE_M: tl.constexpr"]
    )

    lines = [
        "import triton",
        "import triton.language as tl",
        "",
        "",
        f"@triton.autotune(configs={AUTOTUNE_CONFIGS}, key=['M', 'N', 'K'])",
        "@triton.jit",
        f"def {name}(\n    {params},\n):",
        "    pid = tl.program_id(0)",
        "    num_pid_m = tl.cdiv(M, BLOCK_M)",
        "    num_pid_n = tl.cdiv(N, BLOCK_N)",
        "    num_pid_in_group = GROUP_SIZE_M * num_pid_n",
        "    first_pid_m = (pid // num_pid_in_group) * GROUP_SIZE_M",
        "    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)",
        "    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m",
        "    pid_n = (pid % num_pid_in_group) // group_size_m",
        "",
        "    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
        "    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
        "    mask_m = offs_m < M",
        "    mask_n = offs_n < N",
        "",
        "    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)",
        "    for k in range(0, tl.cdiv(K, BLOCK_K)):",
        "        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)",
        "        mask_k = offs_k < K",
        "        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,",
        "                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)",
        "        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,",
        "                    mask=mask_k[:, None] & mask_n[None, :], other=0.0)",
        f"        acc += tl.dot(a, b{dot_args})",
        "",
        "    # ---- epilogue: runs on the tile while it is still in registers ----",
        "    mask = mask_m[:, None] & mask_n[None, :]",
    ]

    names: dict[fx.Node, str] = {}
    for i, inp in enumerate(inputs):
        lines.append(
            f"    x{i} = tl.load(in{i}_ptr + offs_m[:, None] * in{i}_sm"
            f" + offs_n[None, :] * in{i}_sn, mask=mask).to(tl.float32)"
        )
        names[inp] = f"x{i}"

    if matmul.target == aten.addmm.default:
        lines.append(f"    acc = acc + {names[matmul.args[0]]}  # addmm bias")
    names[matmul] = "acc"

    result = emit_pointwise_ops(group, names, lines)
    lines.append(
        f"    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,"
        f" {result}.to({TL_DTYPE[out_dtype]}), mask=mask)"
    )
    return "\n".join(lines) + "\n"


def build_epilogue_op(group, matmul, inputs):
    name = new_kernel_name("gemm_epilogue")
    out_meta = _tensor_meta(group[-1])
    shape, dtype = tuple(out_meta.shape), out_meta.dtype
    source = generate_kernel_source(name, group, matmul, inputs)
    kernel = load_kernel(name, source)

    # ops = " -> ".join(str(n.target) for n in [matmul] + group)
    # print(f"\n--- {name}: {ops}\n{source}")

    def epilogue_op(a, b, *tensors):
        M, K = a.shape
        N = b.shape[1]
        out = torch.empty(shape, dtype=dtype, device=a.device)
        in_strides = [s for t in tensors for s in t.expand(shape).stride()]
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )
        kernel[grid](
            a, b, out, *tensors, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            out.stride(0), out.stride(1), *in_strides,
        )
        return out

    epilogue_op.__name__ = name
    return epilogue_op


def _trial_tensor(val):
    """A real tensor with the shape, dtype and layout of a node's fake value.
    Has its own generator so compiling doesn't change the user's random stream."""
    t = torch.empty_strided(tuple(val.shape), val.stride(), dtype=val.dtype, device=val.device)
    if not t.is_floating_point():
        return t.zero_()
    return t.normal_(generator=torch.Generator(val.device).manual_seed(0))


def _median_seconds(fn, warmup=5, reps=20):
    """Wall-clock time, so Python launch cost counts too. The warmup also
    triggers the kernel's autotuning and compilation."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _epilogue_is_faster(group, matmul, inputs, epilogue_op) -> bool:
    """Time the fused kernel against what runs without it: the matmul as a
    plain ATen call (cuBLAS), then the pointwise group on its own. The winner
    depends on dtype and shape; e.g. fp32 without TF32 favours cuBLAS."""
    group_inputs = external_inputs(group)
    mat1, mat2 = matmul.args[-2:]

    # The compiler runs under FakeTensorMode; these need to really execute.
    with _disable_current_modes(), no_python_dispatcher():
        nodes = dict.fromkeys(matmul.all_input_nodes + group_inputs)
        nodes.pop(matmul)
        env = {n: _trial_tensor(_tensor_meta(n)) for n in nodes}
        fused_op = build_fused_op(group, group_inputs) if len(group) > 1 else None

        def run_separate():
            # Mirrors fuse_pointwise, which leaves single ops to ATen.
            values = dict(env)
            values[matmul] = matmul.target(*(env[a] for a in matmul.args))
            if fused_op is not None:
                return fused_op(*(values[n] for n in group_inputs))
            node = group[0]
            args = (values[a] if isinstance(a, fx.Node) else a for a in node.args)
            return node.target(*args, **node.kwargs)

        fused_args = (env[mat1], env[mat2], *(env[n] for n in inputs))
        fused_time = _median_seconds(lambda: epilogue_op(*fused_args))
        separate_time = _median_seconds(run_separate)
    return fused_time < separate_time


def fuse_epilogue(gm: fx.GraphModule) -> fx.GraphModule:
    """Fold each pointwise group that directly consumes a matmul into that
    matmul's epilogue, so the matmul output never goes to memory. Groups
    that don't qualify, or run faster unfused, are left for fuse_pointwise."""
    for group in find_fusion_groups(gm.graph):
        group_inputs = external_inputs(group)
        matmul = find_epilogue_anchor(group, group_inputs)
        if matmul is None:
            continue

        bias = [matmul.args[0]] if matmul.target == aten.addmm.default else []
        inputs = list(dict.fromkeys(bias + [n for n in group_inputs if n is not matmul]))
        mat1, mat2 = matmul.args[-2:]
        epilogue_op = build_epilogue_op(group, matmul, inputs)
        if not _epilogue_is_faster(group, matmul, inputs, epilogue_op):
            continue

        root = group[-1]
        with gm.graph.inserting_before(root):
            new_node = gm.graph.call_function(epilogue_op, (mat1, mat2, *inputs))
        new_node.meta = dict(root.meta)
        root.replace_all_uses_with(new_node)
        for n in reversed(group):  # consumers before producers
            gm.graph.erase_node(n)
        gm.graph.erase_node(matmul)

    gm.graph.lint()
    gm.recompile()
    return gm
