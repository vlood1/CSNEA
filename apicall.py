import os
from openai import OpenAI
import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialising OpenAI client with API key from environment variable (hidden for security)
client = OpenAI(
    api_key= os.getenv("APIKEY")
)

user_input = "What is this document for?"


# Uploading a file to OpenAI for use
with open("OAPA.pdf", "rb") as f:
    file = client.files.create(file=f, purpose="assistants")
    file_id = file.id


# Creating a vector store (RAG) and uploading the file to it
vs = client.vector_stores.create(name="Simple Vector Store")
client.vector_stores.files.create_and_poll(vector_store_id=vs.id, file_id=file_id)


# Going into the vectore store in order to query it further
results = client.vector_stores.search(
        vector_store_id=vs.id,
        query=user_input, # Query line to search for vector
        rewrite_query=True,
        max_num_results=10
    )

print(results)

def extract_text(results):
    return "\n".join(
        chunk.text
        for result in results.data
        for chunk in result.content
    )

# MVP = Minimal Viable Product

messages = [{"role": "system", "content": "You are a helpful legal assistant. Use the provided legal document context to answer the user's question."}] # Content of the system message - can be modified to change how the model will answer
messages.append({"role": "user", "content": f"Context:\n{(extract_text(results))}\n\nQuestion: {user_input}"})

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
    )

print(response.choices[0].message.content)