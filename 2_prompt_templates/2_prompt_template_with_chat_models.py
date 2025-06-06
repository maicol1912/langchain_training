from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr
import os
from langchain.prompts import ChatPromptTemplate

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

#PART 1: Create ChatPromptTemplate using a template string
print("----Prompt from template----")
template = "Tell me a joke about {topic}"
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"topic":"cats"})
print(f"prompt {prompt}")
result = model.invoke(prompt)
print(f"result: {result.content}")

#PART 2 Prompt with multiples placeholders
print("\n----Prompt with multiples placeholders----")
tamplate_multiple = """You are a helpful assistant.
Human: Tell me a {adjective} short story about a {animal}.
Assistant:"""
prompt_multiple = ChatPromptTemplate.from_template(tamplate_multiple)
prompt = prompt_multiple.invoke({"adjective":"funny","animal":"panda"})

result = model.invoke(prompt)
print(result.content)

#Part 3: Prompt 
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
result = model.invoke(prompt)
print(result.content)