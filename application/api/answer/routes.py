import asyncio
import os
from xml.dom import ValidationErr
from application.core.utility import dict_to_validated_model
from application.models.api_request.answer_api_request import AnswerApiRequest
from application.models.prompt.prompt_resolver import PromptTemplateResolver
from flask import Blueprint, request, Response
import json
import datetime
import logging
import traceback

from pymongo import MongoClient
from bson.objectid import ObjectId
from transformers import GPT2TokenizerFast
from langchain_openai import OpenAIEmbeddings
from langchain.utils.math import cosine_similarity

from langchain_core.prompts import PromptTemplate


from application.core.settings import settings
from application.models.vectorstore.vector_creator import VectorCreator
from application.models.llm.llm_creator import LLMCreator
from application.error import bad_request
from flask import jsonify
from pydantic import ValidationError


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

# load the prompts
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(current_dir, "prompts", "chat_combine_default.txt"), "r") as f:
    chat_combine_template = f.read()

with open(os.path.join(current_dir, "prompts", "chat_reduce_prompt.txt"), "r") as f:
    chat_reduce_template = f.read()

with open(os.path.join(current_dir, "prompts", "chat_combine_creative.txt"), "r") as f:
    chat_combine_creative = f.read()

with open(os.path.join(current_dir, "prompts", "chat_combine_strict.txt"), "r") as f:
    chat_combine_strict = f.read()    

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


def complete_stream(question, docsearch, chat_history, api_key, prompt_id, conversation_id):
    llm = LLMCreator.create_llm(settings.LLM_NAME, api_key=api_key)
    # TODO: No way we cant search on a collection of docs. should look deeper
    docs = docsearch.search(question, k=2)
    if settings.LLM_NAME == "llama.cpp":
        docs = [docs[0]]
    # join all page_content together with a newline
    docs_together = "\n".join([doc.page_content for doc in docs])

    # We need to be doing some real chain lang shit here
    # - any of the cookbooks but that will require skipping below
    prompt = PromptTemplateResolver.resolve()
    p_chat_combine = prompt.format(summaries=docs_together)
    messages_combine = [{"role": "system", "content": p_chat_combine}]
    source_log_docs = []

    # Conversation model/ interface 
    for doc in docs:
        if doc.metadata:
            data = json.dumps({"type": "source", "doc": doc.page_content[:10], "metadata": doc.metadata})
            source_log_docs.append({"title": doc.metadata['title'].split('/')[-1], "text": doc.page_content})
        else:
            data = json.dumps({"type": "source", "doc": doc.page_content[:10]})
            source_log_docs.append({"title": doc.page_content, "text": doc.page_content})
        yield f"data:{data}\n\n"

    if len(chat_history) > 1:
        tokens_current_history = 0
        # count tokens in history
        chat_history.reverse()
        for i in chat_history:
            if "prompt" in i and "response" in i:
                tokens_batch = count_tokens(i["prompt"]) + count_tokens(i["response"])
                if tokens_current_history + tokens_batch < settings.TOKENS_MAX_HISTORY:
                    tokens_current_history += tokens_batch
                    messages_combine.append({"role": "user", "content": i["prompt"]})
                    messages_combine.append({"role": "system", "content": i["response"]})
    messages_combine.append({"role": "user", "content": question})

    response_full = ""
    completion = llm.gen_stream(model=gpt_model, engine=settings.AZURE_DEPLOYMENT_NAME,
                                messages=messages_combine)
    for line in completion:
        data = json.dumps({"answer": str(line)})
        response_full += str(line)
        yield f"data: {data}\n\n"

    # save conversation to database
    if conversation_id is not None:
        conversations_collection.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$push": {"queries": {"prompt": question, "response": response_full, "sources": source_log_docs}}},
        )

    else:
        # create new conversation
        # generate summary
        messages_summary = [{"role": "assistant", "content": "Summarise following conversation in no more than 3 "
                                                             "words, respond ONLY with the summary, use the same "
                                                             "language as the system \n\nUser: " + question + "\n\n" +
                                                             "AI: " +
                                                             response_full},
                            {"role": "user", "content": "Summarise following conversation in no more than 3 words, "
                                                        "respond ONLY with the summary, use the same language as the "
                                                        "system"}]

        completion = llm.gen(model=gpt_model, engine=settings.AZURE_DEPLOYMENT_NAME,
                             messages=messages_summary, max_tokens=30)
        conversation_id = conversations_collection.insert_one(
            {"user": "local",
             "date": datetime.datetime.utcnow(),
             "name": completion,
             "queries": [{"prompt": question, "response": response_full, "sources": source_log_docs}]}
        ).inserted_id

    # send data.type = "end" to indicate that the stream has ended as json
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
        llm = LLMCreator.create_llm(settings.LLM_NAME, api_key=data.api_key)

        docs = docsearch.search(data.question, k=2)
        prompt = PromptTemplateResolver.resolve()
        # join all page_content together with a newline
        docs_together = "\n".join([doc.page_content for doc in docs])
        p_chat_combine = prompt.replace("{summaries}", docs_together)
        messages_combine = [{"role": "system", "content": p_chat_combine}]
        source_log_docs = []
        for doc in docs:
            if doc.metadata:
                source_log_docs.append({"title": doc.metadata['title'].split('/')[-1], "text": doc.page_content})
            else:
                source_log_docs.append({"title": doc.page_content, "text": doc.page_content})
        # join all page_content together with a newline

        # TODO: This should be just pulling history based on conversation_id.
        if len(data.history) > 1:
            tokens_current_history = 0
            # count tokens in history
            data.history.reverse()
            for i in data.history:
                if "prompt" in i and "response" in i:
                    tokens_batch = count_tokens(i["prompt"]) + count_tokens(i["response"])
                    if tokens_current_history + tokens_batch < settings.TOKENS_MAX_HISTORY:
                        tokens_current_history += tokens_batch
                        messages_combine.append({"role": "user", "content": i["prompt"]})
                        messages_combine.append({"role": "system", "content": i["response"]})
        messages_combine.append({"role": "user", "content": data.question})


        completion = llm.gen(model=gpt_model, engine=settings.AZURE_DEPLOYMENT_NAME,
                                    messages=messages_combine)


        result = {"answer": completion, "sources": source_log_docs}
        logger.debug(result)

        # generate conversationId
        if conversation_id is not None:
            conversations_collection.update_one(
                {"_id": ObjectId(conversation_id)},
                {"$push": {"queries": {"prompt": data.question,
                                       "response": result["answer"], "sources": result['sources']}}},
            )

        else:
            # create new conversation
            # generate summary
            messages_summary = [
                {"role": "assistant", "content": "Summarise following conversation in no more than 3 words, "
                    "respond ONLY with the summary, use the same language as the system \n\n"
                    "User: " + data.question + "\n\n" + "AI: " + result["answer"]},
                {"role": "user", "content": "Summarise following conversation in no more than 3 words, "
                    "respond ONLY with the summary, use the same language as the system"}
            ]

            completion = llm.gen(
                model=gpt_model,
                engine=settings.AZURE_DEPLOYMENT_NAME,
                messages=messages_summary,
                max_tokens=30
            )
            conversation_id = conversations_collection.insert_one(
                {"user": "local",
                "date": datetime.datetime.utcnow(),
                "name": completion,
                "queries": [{"prompt": data.question, "response": result["answer"], "sources": source_log_docs}]}
            ).inserted_id

        result["conversation_id"] = str(conversation_id)

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
