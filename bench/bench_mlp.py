import torch
import triton

from models.mlp import MLPBlock
from compiler import my_backend


if __name__ == "__main__":
    
    torch.manual_seed(0)
    model = MLPBlock().cuda()
    x = torch.randn(512, 1024, device="cuda")
    
    inductor_compiled = torch.compile(model, backend="inductor")
    my_compiled = torch.compile(model, backend=my_backend)
    
    with torch.no_grad():
        expected = model(x)
        actual = my_compiled(x)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    print("compiled model output matches eager.")

    # # Training: forward and backward graphs both go through the fusion pass.
    # torch._dynamo.reset()
    # compiled = torch.compile(model, backend=fusion_backend(verbose=False))
    # compiled(x).sum().backward()
    # fused_grad = model.fc1.weight.grad.clone()
    # model.zero_grad()
    # model(x).sum().backward()
    # torch.testing.assert_close(fused_grad, model.fc1.weight.grad, rtol=1e-3, atol=1e-3)
    # print("Fused gradients match eager.")

    with torch.no_grad():
        eager_ms = triton.testing.do_bench(lambda: model(x))
        inductor_ms = triton.testing.do_bench(lambda: inductor_compiled(x))
        fusion_ms = triton.testing.do_bench(lambda: my_compiled(x))
        
    print(f"eager: {eager_ms:.3f} ms\n"
          f"inductor: {inductor_ms:.3f} ms\n"
          f"fusion: {fusion_ms:.3f} ms")