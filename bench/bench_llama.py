import torch
from transformers import AutoConfig, LlamaConfig, LlamaForCausalLM
import triton 
from compiler.backend import custom_backend


config = LlamaConfig(
    vocab_size=32000,
    hidden_size=3072,
    intermediate_size=8192,
    num_hidden_layers=28,
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=4096,
    rms_norm_eps=1e-05,
    rope_theta=10000.0,
)


if __name__ == "__main__":
    
    with torch.device("cuda"):
        eager_model = LlamaForCausalLM(config).to(torch.float16)
        
    compiled_model = torch.compile(eager_model, backend=custom_backend)
    inductor_model = torch.compile(eager_model, backend="inductor")
    
    batch_size = 1
    sequence_length = 512
    input_ids = torch.randint(
        low=0, 
        high=config.vocab_size, 
        size=(batch_size, sequence_length), 
        device="cuda"
    )

    with torch.no_grad():
        eager_ms = triton.testing.do_bench(lambda: eager_model(input_ids))
        compiled_ms = triton.testing.do_bench(lambda: compiled_model(input_ids))
        inductor_ms = triton.testing.do_bench(lambda: inductor_model(input_ids))
        
        print(f"\neager: {eager_ms:.3f} ms"
              f"\ncompiled: {compiled_ms:.3f} ms"
              f"\ninductor: {inductor_ms:.3f} ms")

    
