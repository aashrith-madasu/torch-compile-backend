import torch
import triton
import argparse

from mlp import MLPBlock
from compiler.backend import custom_backend


precision_to_dtype = {
    "16": torch.float16,
    "32": torch.float32,
    "64": torch.float64
}

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", type=str)
    args = parser.parse_args()
    precision = args.precision
    
    dtype = precision_to_dtype[precision]
    eager_model = MLPBlock().to(device="cuda", dtype=dtype)
    inductor_model = torch.compile(eager_model, backend="inductor")
    custom_model = torch.compile(eager_model, backend=custom_backend)
    
    x = torch.randn(512, 1024).to(device="cuda", dtype=dtype)
    

    with torch.no_grad():
        torch.testing.assert_close(custom_model(x), eager_model(x), rtol=1e-4, atol=1e-4,
                                    msg="assert failed: outputs did not match with eager")

    custom_model(x).sum().backward()
    custom_grad = custom_model.fc1.weight.grad.clone()
    eager_model.zero_grad()
    eager_model(x).sum().backward()
    expected_grad = eager_model.fc1.weight.grad.clone()
    torch.testing.assert_close(custom_grad, expected_grad, rtol=1e-3, atol=1e-3,
                                msg="assert failed: gradients did not match with eager")


    with torch.no_grad():
        inductor_ms = triton.testing.do_bench(lambda: inductor_model(x))
        custom_ms = triton.testing.do_bench(lambda: custom_model(x))
        
        print(f"inductor: {inductor_ms:.3f} ms\n"
              f"custom backend: {custom_ms:.3f} ms")