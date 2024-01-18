from application.core.settings import settings
from application.models.llm.base import BaseLLM
from langchain.llms import HuggingFacePipeline

class HuggingFaceLLM(BaseLLM):

    def __init__(self, api_key, llm_name=settings.HUGGING_FACE_REPO_NAME,q=False):
        if q:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
            tokenizer = AutoTokenizer.from_pretrained(llm_name)
            bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16
                        )
            model = AutoModelForCausalLM.from_pretrained(llm_name,quantization_config=bnb_config)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            tokenizer = AutoTokenizer.from_pretrained(llm_name)
            model = AutoModelForCausalLM.from_pretrained(llm_name)
        
        pipe = pipeline(
            "text-generation", model=model,
            tokenizer=tokenizer, max_new_tokens=2000,
            device_map="auto", eos_token_id=tokenizer.eos_token_id
        )
        self.model = HuggingFacePipeline(pipeline=pipe)

    def gen(self, model, engine, messages, stream=False, **kwargs):
        context = messages[0]['content']
        user_question = messages[-1]['content']
        prompt = f"### Instruction \n {user_question} \n ### Context \n {context} \n ### Answer \n"

        result = self.model(prompt)

        return result.content

    def gen_stream(self, model, engine, messages, stream=True, **kwargs):

        raise NotImplementedError("HuggingFaceLLM Streaming is not implemented yet.")

