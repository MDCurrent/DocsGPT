import asyncio
from application.core.utility import dict_to_validated_model
from application.models.api_request.answer_api_request import AnswerApiRequest
from application.models.chain.chain_creator import ChainCreator
from flask import Blueprint, request, Response
from flask import jsonify
from pydantic import ValidationError
import json
import logging
import traceback

from pymongo import MongoClient
from bson.objectid import ObjectId
from transformers import GPT2TokenizerFast
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from application.core.settings import settings
from application.models.vectorstore.vector_creator import VectorCreator
from application.models.llm.llm_creator import LLMCreator
from application.error import bad_request



logger = logging.getLogger(__name__)

mongo = MongoClient(settings.MONGO_URI)
db = mongo["docsgpt"]
conversations_collection = db["conversations"]
vectors_collection = db["vectors"]
prompts_collection = db["prompts"]
answer = Blueprint('answer', __name__)

if settings.LLM_NAME == "gpt4":
    gpt_model = 'gpt-4'
elif settings.LLM_NAME == "anthropic":
    gpt_model = 'claude-2'
else:
    gpt_model = 'gpt-3.5-turbo'

api_key_set = settings.API_KEY is not None
embeddings_key_set = settings.EMBEDDINGS_KEY is not None

async def async_generate(chain, question, chat_history):
    result = await chain.arun({"question": question, "chat_history": chat_history})
    return result


def count_tokens(string):
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    return len(tokenizer(string)['input_ids'])


def run_async_chain(chain, question, chat_history):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = {}
    try:
        answer = loop.run_until_complete(async_generate(chain, question, chat_history))
    finally:
        loop.close()
    result["answer"] = answer
    return result

def is_azure_configured():
    return settings.OPENAI_API_BASE and settings.OPENAI_API_VERSION and settings.AZURE_DEPLOYMENT_NAME


def complete_stream(prompt, docsearch, api_key, conversation_id):
    model = LLMCreator.create_llm(settings.LLM_NAME, api_key=api_key)
    # TODO: https://python.langchain.com/docs/integrations/vectorstores/faiss#merging
    # find relevant context
    docs = docsearch.search(prompt, k=2)
    if settings.LLM_NAME == "llama.cpp":
        docs = [docs[0]]


    chain = ChainCreator.create_chain(model,docs,prompt)
    inputs = {"input": prompt}

    # TODO: Conversation model/ interface - probably not this low level either
    response_full = ""
    async def process_output(chain, input):
        response_full = ""
        async for output in chain.astream(input):
            data = json.dumps({"answer": str(output)})
            response_full += str(answer)
            yield f"data: {data}\n\n"


    # Call the async function
    asyncio.run(process_output(chain, inputs))

    # save conversation to database
    if conversation_id is not None:
        conversations_collection.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$push": {"queries": {"prompt": prompt, "response": response_full, "sources": docs}}},
        )
    # send data.type = "end" to indicate that the stream has ended as json
    # This seems fucked up
    data = json.dumps({"type": "id", "id": str(conversation_id)})
    yield f"data: {data}\n\n"
    data = json.dumps({"type": "end"})
    yield f"data: {data}\n\n"


@answer.route("/stream", methods=["POST"])
def stream():
    try:
        data = dict_to_validated_model(AnswerApiRequest, request.get_json())
    except ValidationError as e:
        return jsonify(error=str(e)), 400
    docsearch = VectorCreator.create_vectorstore(settings.VECTOR_STORE, data.vectorstore, data.embeddings_key)

    return Response(
        complete_stream(data.question, docsearch,
                        chat_history=data.history, api_key=data.api_key,
                        prompt_id=data.prompt_id,
                        conversation_id=data.conversation_id), mimetype="text/event-stream"
    )


@answer.route("/api/answer", methods=["POST"])
def api_answer():
    try:
        data = dict_to_validated_model(AnswerApiRequest, request.get_json())
    except ValidationError as e:
        return jsonify(error=str(e)), 400
    # use try and except  to check for exception
    try:
        # loading the index and the store and the prompt template
        # Note if you have used other embeddings than OpenAI, you need to change the embeddings
        docsearch = VectorCreator.create_vectorstore(settings.VECTOR_STORE, data.vectorstore, data.embeddings_key)

        model = LLMCreator.create_llm(settings.LLM_NAME, api_key=data.api_key)
        # TODO: https://python.langchain.com/docs/integrations/vectorstores/faiss#merging
        # find relevant context
        docs = docsearch.search(data.question, k=2)
        prompt = data.question
        if settings.LLM_NAME == "llama.cpp":
            docs = [docs[0]]


        chain = ChainCreator.create_chain(model,docs,prompt)
        inputs = {"input": prompt}
    
        response = chain.invoke(inputs)


        result = {"answer": response, "sources": docs}
        logger.debug(result)

        # generate conversationId
        if data.conversation_id is not None:
            #TODO: Grab conversation and populate with history
            conversations_collection.update_one(
                {"_id": ObjectId(data.conversation_id)},
                {"$push": {"queries": {"prompt": data.question,
                                       "response": result["answer"], "sources": result['sources']}}},
            )

        result["conversation_id"] = str(data.conversation_id)

        # mock result
        # result = {
        #     "answer": "The answer is 42",
        #     "sources": ["https://en.wikipedia.org/wiki/42_(number)", "https://en.wikipedia.org/wiki/42_(number)"]
        # }
        return result
    except Exception as e:
        # print whole traceback
        traceback.print_exc()
        print(str(e))
        return bad_request(500, str(e))
