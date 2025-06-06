from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr
import os
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain.prompts import ChatPromptTemplate
from typing import Any

load_dotenv()

api_key = os.getenv("OPEN_ROUTER_KEY")
if not api_key:
    raise ValueError("OPEN_ROUTER_KEY no está configurada en el archivo .env")

api_key_secret = SecretStr(api_key)

model = ChatOpenAI(
    model="deepseek/deepseek-r1:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key_secret,
    temperature=0.7,
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

uppercase_output = RunnableLambda(lambda x: x.upper()) # type: ignore
count_words = RunnableLambda(lambda x : f"Word count: {len(x.split())}\n{x}") #type: ignore

chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

result = chain.invoke({"topic":"lawyers","joke_count":3})

print(result)