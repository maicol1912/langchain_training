from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr
import os
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel
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
        ("system", "You are an expert product reviewer"),
        ("human", "List the main features of the product {product_name}"),
    ]
)

def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            ("human", "Given these features: {features} list the pros of these features")
        ]
    )

    return pros_template.format_prompt(features=features)

def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages([
       ("system", "You are an expert product reviewer."),
       ("human", "Given these features: {features}, list the cons of these features")
    ])
    
    return cons_template.format_prompt(features=features)
    
def combine_pros_cons(pros,cons):
    return f"Pros: \n${pros}\nCons:\n{cons}"


pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"])) #type:ignore
)

result = chain.invoke({"product_name":"MacBook Pro"})

print(result)