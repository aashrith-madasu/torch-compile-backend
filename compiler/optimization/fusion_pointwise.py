import importlib.util
import os
import tempfile

import torch
import torch.fx as fx
import triton

aten = torch.ops.aten

# ATen op -> function that builds a Triton expression from argument strings.
POINTWISE_OPS = {
    aten.add.Tensor: lambda a, b: f"({a} + {b})",
    aten.sub.Tensor: lambda a, b: f"({a} - {b})",
    aten.mul.Tensor: lambda a, b: f"({a} * {b})",
    aten.div.Tensor: lambda a, b: f"({a} / {b})",
    aten.neg.default: lambda a: f"(-{a})",
    aten.abs.default: lambda a: f"tl.abs({a})",
    aten.relu.default: lambda a: f"tl.maximum({a}, 0.0)",
    aten.exp.default: lambda a: f"tl.exp({a})",
    aten.log.default: lambda a: f"tl.log({a})",
    aten.sqrt.default: lambda a: f"tl.sqrt({a})",
    aten.rsqrt.default: lambda a: f"tl.rsqrt({a})",
    aten.erf.default: lambda a: f"tl.erf({a})",
    aten.sigmoid.default: lambda a: f"tl.sigmoid({a})",
    aten.tanh.default: lambda a: f"(2.0 * tl.sigmoid(2.0 * {a}) - 1.0)",
    aten.maximum.default: lambda a, b: f"tl.maximum({a}, {b})",
    aten.minimum.default: lambda a, b: f"tl.minimum({a}, {b})",
    aten._to_copy.default: lambda a: a,
}

# Composite ops we break down into the primitives above so they can fuse.


KERNEL_DIR = tempfile.mkdtemp(prefix="fused_kernels_")
_kernel_counter = 0


def _tensor_meta(node):
    val = node.meta.get("val") if isinstance(node, fx.Node) else None
    return val if isinstance(val, torch.Tensor) else None


def _is_float_cast(node: fx.Node) -> bool:
    """A plain dtype conversion between floats that fit in fp32. A cast below
    fp32 must end its group: only tl.store actually rounds to the output
    dtype, so ops after it in the kernel would see the unrounded value."""
    src, dst = _tensor_meta(node.args[0]), _tensor_meta(node)
    return (
        set(node.kwargs) == {"dtype"}
        and src is not None
        and dst is not None
        and src.is_floating_point()
        and torch.float64 not in (src.dtype, dst.dtype)
        and (dst.dtype == torch.float32 or not any(_is_fusible(u) for u in node.users))
    )


def _is_fusible(node: fx.Node) -> bool:
    if node.op != "call_function" or node.target not in POINTWISE_OPS:
        return False
    if node.target == aten._to_copy.default:
        if not _is_float_cast(node):
            return False
    elif node.kwargs:
        return False
    val = _tensor_meta(node)
    return (
        val is not None
        and val.is_cuda
        and val.is_floating_point()
        and all(isinstance(s, int) for s in val.shape)  # static shapes only
    )


def find_fusion_groups(graph: fx.Graph) -> list[list[fx.Node]]:
    """Walk the graph backwards (consumers before producers) and pull each
    pointwise node into its consumers' group when
    (a) all of its users are in that one group, and
    (b) it has the same output shape as the group.

    (a) keeps every group single-output: only the group's last node is read
    outside it, so no intermediate value ever needs to be written to memory.
    A node may still have several users (gelu reads its input twice) as long
    as they all end up in the same kernel.
    """
    group_of: dict[fx.Node, list[fx.Node]] = {}
    for node in reversed(graph.nodes):
        if not _is_fusible(node):
            continue
        consumer_groups = [group_of.get(u) for u in node.users]
        group = consumer_groups[0] if consumer_groups else None
        if (
            group is not None
            and all(g is group for g in consumer_groups)
            and _tensor_meta(node).shape == _tensor_meta(group[0]).shape
        ):
            group.append(node)
        else:
            group = [node]
        group_of[node] = group

    order = {n: i for i, n in enumerate(graph.nodes)}
    groups = {id(g): g for g in group_of.values()}.values()
    return [sorted(g, key=order.__getitem__) for g in groups]


def external_inputs(group: list[fx.Node]) -> list[fx.Node]:
    """Nodes the group reads from outside, deduplicated, in first-use order."""
    members = set(group)
    return list(dict.fromkeys(
        arg for node in group for arg in node.all_input_nodes if arg not in members
    ))


def emit_pointwise_ops(group, names, lines) -> str:
    """Append one Triton line per node in `group` and return the variable
    holding the group's result. `names` maps each external input node to the
    Triton variable it was loaded into."""
    def operand(arg):
        if isinstance(arg, fx.Node):
            return names[arg]
        return repr(float(arg))  # Python scalar baked in as a constant

    for j, node in enumerate(group):
        expr = POINTWISE_OPS[node.target](*map(operand, node.args))
        lines.append(f"    t{j} = {expr}  # {node.target}")
        names[node] = f"t{j}"
    return names[group[-1]]


def generate_kernel_source(name, group, inputs, rank):
    """Elementwise kernel over the flattened output. Each input is indexed
    with its own strides (stride 0 on broadcast dims), so broadcasting and
    non-contiguous inputs work without copies."""
    in_ptrs = [f"in{i}_ptr" for i in range(len(inputs))]
    sizes = [f"size{d}" for d in range(rank)]
    strides = [f"in{i}_s{d}" for i in range(len(inputs)) for d in range(rank)]
    params = ", ".join(in_ptrs + ["out_ptr", "numel"] + sizes + strides + ["BLOCK: tl.constexpr"])

    lines = [
        "import triton",
        "import triton.language as tl",
        "",
        "",
        "@triton.jit",
        f"def {name}({params}):",
        "    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
        "    mask = offsets < numel",
    ]

    # Unravel the flat output index into per-dimension indices.
    lines.append("    rem = offsets")
    for d in reversed(range(rank)):
        if d == 0:
            lines.append("    idx0 = rem")
        else:
            lines.append(f"    idx{d} = rem % size{d}")
            lines.append(f"    rem = rem // size{d}")

    names: dict[fx.Node, str] = {}
    for i, inp in enumerate(inputs):
        offset = " + ".join(f"idx{d} * in{i}_s{d}" for d in range(rank)) or "0"
        lines.append(f"    x{i} = tl.load(in{i}_ptr + {offset}, mask=mask).to(tl.float32)")
        names[inp] = f"x{i}"

    result = emit_pointwise_ops(group, names, lines)
    lines.append(f"    tl.store(out_ptr + offsets, {result}, mask=mask)")
    return "\n".join(lines) + "\n"


def new_kernel_name(prefix):
    global _kernel_counter
    _kernel_counter += 1
    return f"{prefix}_{_kernel_counter - 1}"


def load_kernel(name, source):
    # triton.jit needs the source to live in a real file (it uses inspect).
    path = os.path.join(KERNEL_DIR, f"{name}.py")
    with open(path, "w") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def build_fused_op(group, inputs):
    name = new_kernel_name("fused")

    out_meta = _tensor_meta(group[-1])
    shape, dtype = tuple(out_meta.shape), out_meta.dtype
    rank = len(shape)
    source = generate_kernel_source(name, group, inputs, rank)
    kernel = load_kernel(name, source)

    # ops = " -> ".join(str(n.target) for n in group)
    # print(f"\n--- {name}: {ops}\n{source}")

    def fused_op(*tensors):
        out = torch.empty(shape, dtype=dtype, device=tensors[0].device)
        strides = [s for t in tensors for s in t.expand(shape).stride()]
        numel = out.numel()
        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK"]),)
        kernel[grid](*tensors, out, numel, *shape, *strides, BLOCK=1024)
        return out

    fused_op.__name__ = name
    return fused_op


def fuse_pointwise(gm: fx.GraphModule) -> fx.GraphModule:
    groups = find_fusion_groups(gm.graph)
    for group in groups:
        if len(group) < 2:  # fusing a single op gains nothing; leave it to ATen
            continue
        inputs = external_inputs(group)
        fused_op = build_fused_op(group, inputs)

        root = group[-1]
        with gm.graph.inserting_before(root):
            new_node = gm.graph.call_function(fused_op, tuple(inputs))
        new_node.meta = dict(root.meta)
        root.replace_all_uses_with(new_node)
        for n in reversed(group):  # consumers before producers
            gm.graph.erase_node(n)

    gm.graph.lint()
    gm.recompile()
    return gm