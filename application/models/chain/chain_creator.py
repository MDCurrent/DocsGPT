class ChainCreator:


    @classmethod
    def create_chain(cls, model, docs, prompt):
        # we should load the convo or create a new one and deprecate the api chat history
        # - any of the cookbooks but that will require skipping below
        prompt_template = PromptTemplateResolver.resolve(prompt)
        prompt_template.format(summaries=str(docs))
        memory = ConversationBufferMemory(return_messages=True)
        return (
            RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt_template
        | model
        )