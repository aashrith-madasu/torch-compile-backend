import timm
import torch
import triton 
from compiler.backend import custom_backend


if __name__ == "__main__":
    
    with torch.device("cuda"):
        eager_model = timm.create_model('vit_base_patch16_224', pretrained=False).to(dtype=torch.float16)
        x = torch.randn(1, 3, 224, 224).to(dtype=torch.float16)


    compiled_model = torch.compile(eager_model, backend=custom_backend)
    inductor_model = torch.compile(eager_model, backend="inductor")
    
   
    with torch.no_grad():
        eager_ms = triton.testing.do_bench(lambda: eager_model(x))
        compiled_ms = triton.testing.do_bench(lambda: compiled_model(x))
        inductor_ms = triton.testing.do_bench(lambda: inductor_model(x))
        
        print(f"\neager: {eager_ms:.3f} ms"
              f"\ncompiled: {compiled_ms:.3f} ms"
              f"\ninductor: {inductor_ms:.3f} ms")
        