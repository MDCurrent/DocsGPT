import torch
from vllm import LLM, SamplingParams
mname = "mistralai/Mistral-7B-Instruct-v0.1"

llm = LLM(
     model=mname,
     tokenizer=mname,
     tokenizer_mode="auto",
     trust_remote_code=True,
     tensor_parallel_size=1,
     dtype=torch.float16,
     quantization=None,
     seed=42,
     gpu_memory_utilization=0.8,
     swap_space=0,
     enforce_eager=False,
     max_context_len_to_capture=1024,
 )