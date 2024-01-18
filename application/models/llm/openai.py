from application.models.llm.base import BaseLLM
from application.core.settings import settings
from langchain_openai import OpenAI,AzureOpenAI, AzureOpenAI

class OpenAILLM(BaseLLM):

    def __init__(self, api_key):
        global langchain_openai
        
        self.client = OpenAI(
                api_key=api_key, 
            )
        self.api_key = api_key

    def _get_openai(self):
        # Import openai when needed
        import langchain_openai
        
        return langchain_openai

    def gen(self, model, engine, messages, stream=False, **kwargs):
        response = self.client.chat.completions.create(model=model,
            messages=messages,
            stream=stream,
            **kwargs)

        return response.choices[0].message.content

    def gen_stream(self, model, engine, messages, stream=True, **kwargs):
        response = self.client.chat.completions.create(model=model,
            messages=messages,
            stream=stream,
            **kwargs)

        for line in response:
            # import sys
            # print(line.choices[0].delta.content, file=sys.stderr)
            if line.choices[0].delta.content is not None:
                yield line.choices[0].delta.content


class AzureOpenAILLM(OpenAILLM):
    """
    https://python.langchain.com/docs/integrations/llms/azure_openai#api-configuration
    """

    def __init__(self, openai_api_key, openai_api_base, openai_api_version, deployment_name):
        super().__init__(openai_api_key)
        import os
        from azure.identity import DefaultAzureCredential

        # Get the Azure Credential
        credential = DefaultAzureCredential()

        # Set the API type to `azure_ad`
        os.environ["OPENAI_API_TYPE"] = "azure_ad"
        # Set the API_KEY to the token from the Azure credential
        os.environ["OPENAI_API_KEY"] = credential.get_token("https://cognitiveservices.azure.com/.default").token
        self.api_base = settings.OPENAI_API_BASE,
        self.api_version = settings.OPENAI_API_VERSION,
        self.deployment_name = settings.AZURE_DEPLOYMENT_NAME,

        self.client = AzureOpenAI(
            api_key=openai_api_key,  
            api_version=settings.OPENAI_API_VERSION,
            api_base=settings.OPENAI_API_BASE,
            deployment_name=settings.AZURE_DEPLOYMENT_NAME,
        )

    def _get_openai(self):
        openai = super()._get_openai()

        return openai
