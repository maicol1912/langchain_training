from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from pydantic import SecretStr
import os

load_dotenv()

api_key = os.getenv("OPEN_ROUTER_KEY")
if not api_key:
    raise ValueError("OPEN_ROUTER_KEY no está configurada en el archivo .env")

api_key_secret = SecretStr(api_key)

llm = ChatOpenAI(
    model="deepseek/deepseek-r1:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key_secret,
    temperature=0.7,
)

messages = [
    SystemMessage(content="solve the following math problems"),
    HumanMessage(content="what is 81 divided by 9?"),
]

messages = [
    SystemMessage(content="solve the following math problems"),
    HumanMessage(content="what is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9"),
    HumanMessage(content="what is 10 times 5?"),
]
response = llm.invoke(messages)
print(f"Answer from IA: {response.content}")