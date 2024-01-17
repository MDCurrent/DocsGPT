from langchain_openai import OpenAIEmbeddings
from langchain.utils.math import cosine_similarity

from langchain_core.prompts import PromptTemplate

import glob


class PromptTemplateResolver:

    def __init__(self):
        self.templates = self.load_templates()
    def load_templates(self):
        templates = []
        files = glob.glob('application/prompts/*.txt')  # adjust the pattern to match your files
        for file in files:
            with open(file, 'r') as f:
                template = PromptTemplate.from_template(f.read())
                templates.append(template)
        return templates        


    def resolve(self, input):
        embeddings_model = OpenAIEmbeddings()
        query_embedding = embeddings_model.embed_query(input["query"])
        similarity = cosine_similarity([query_embedding], query_embedding)[0]
        most_similar = self.templates[similarity.argmax()]
        print(f"most simmilar = {most_similar}")
        return PromptTemplate.from_template(most_similar)

