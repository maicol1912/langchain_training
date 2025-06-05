from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from pydantic import SecretStr
import os
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID_FIRESTORE")
SESSION_ID = "user1_session"
COLLECTION_NAME = "chat_history"

api_key = os.getenv("OPEN_ROUTER_KEY")
if not api_key:
    raise ValueError("OPEN_ROUTER_KEY no está configurada en el archivo .env")

api_key_secret = SecretStr(api_key)

print("Initializing Firestore Client...")
firestore_client = firestore.Client(project=PROJECT_ID)

print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=firestore_client
)
print("chat history initialized")
print("current Chat history: ", chat_history.messages)

model = ChatOpenAI(
    model="deepseek/deepseek-r1:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key_secret,
    temperature=0.7,
)
print("Start chatting with the AI. Type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if(human_input.lower() == "exit"):
        break
    
    chat_history.add_user_message(human_input)
    
    ai_response = model.invoke(chat_history.messages)
    
    chat_history.add_ai_message(str(ai_response.content))
    
    print(f"AI: {ai_response.content}")