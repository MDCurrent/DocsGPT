from application.models.llm.sagemaker import SagemakerAPILLM
from application.models.llm.docsgpt_provider import DocsGPTAPILLM
from langchain.llms import HuggingFacePipeline,OpenAI,AzureOpenAI, AzureOpenAI, LlamaCpp, ChatAnthropic, DocsGPTAPILLM


class LLMCreator:
    llms = {
        'openai': OpenAI,
        'azure_openai': AzureOpenAI,
        'sagemaker': SagemakerAPILLM,
        'huggingface': HuggingFacePipeline,
        'llama.cpp': LlamaCpp,
        'anthropic': ChatAnthropic,
        'docsgpt': DocsGPTAPILLM
    }

    @classmethod
    def create_llm(cls, type, *args, **kwargs):
        llm_class = cls.llms.get(type.lower())
        if not llm_class:
            raise ValueError(f"No LLM class found for type {type}")
        return llm_class(*args, **kwargs)