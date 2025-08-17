
from typing import AsyncGenerator
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from settings import settings
from .db import get_vector_db

LLM_MODEL = settings.LLM_MODEL


def get_prompt_template():
    QUERY_PROMPT = PromptTemplate(
        input_variable=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}"""
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    return QUERY_PROMPT, prompt


async def query(input: str) -> AsyncGenerator[str, None]:
    if input is None or input.strip() == "":
        return None

    llm = ChatOllama(model=LLM_MODEL)
    db = get_vector_db()
    query_prompt, prompt = get_prompt_template()

    retriever = MultiQueryRetriever.from_llm(
        db.as_retriever(search_kwargs={'k': 5}),
        llm=llm,
        prompt=query_prompt
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    async for chunk in chain.astream(input):
        if "result" in chunk:
            print(chunk["result"])
            yield chunk["result"]
