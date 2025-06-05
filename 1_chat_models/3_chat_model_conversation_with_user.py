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

model = ChatOpenAI(
    model="deepseek/deepseek-r1:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key_secret,
    temperature=0.7,
)

chat_history = []
system_message = SystemMessage(content="you are a helpful Ai assitant.")
chat_history.append(system_message)

while True:
    query = input("You: ")
    if(query.lower() == "exit"):
        break
    chat_history.append(HumanMessage(content=query))
    
    result = model.invoke(chat_history)
    
    response = result.content
    chat_history.append(AIMessage(content=response))
    
    print(f"AI: {response}")
    
print("-----Message history-------")
print(chat_history)