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

user_input = "What punishment would I get for murdering someone?" # Example user input question


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

# print(results)

print(results.model_dump())

# def extract_text(results):
#     return "\n".join(
#         chunk.text
#         for result in results.data
#         for chunk in result.content
#     )

# Method to extract and place together text content from results that have a score above 0.5
def getobj(results):
    objlist = []
    for i in results.data:
        if i.score > 0.5:
            objlist.append("\n".join([chunk.text for chunk in i.content]))

    return objlist

context = "\n".join([text for text in getobj(results)])


# MVP = Minimal Viable Product

messages = [{"role": "system", "content": "You are a helpful legal assistant. Use the provided legal document context to answer the user's question. Also, please cite the sections where you have retrieved relevant information along with answering the user's query."}] # Content of the system message - can be modified to change how the model will answer
messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"})


# Response creation
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
    )

# ADD WAY TO REMEMBER THINGS! (messages list) ^^^

print(response.choices[0].message.content)



