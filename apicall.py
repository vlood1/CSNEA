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

# Constants / Configuration
OPENAI_MODEL = "gpt-4o-mini"
MAX_HISTORY_LENGTH = 6

# Uploading a file to OpenAI for use

def file_upload(filename):
    with open(filename, "rb") as f:
        file = client.files.create(file=f, purpose="assistants")
        print(f"[✓] Uploaded {filename} → ID: {file.id}")
        return file.id




# Creating a vector store (RAG) and uploading the file to it
def create_vector_store(file_id):
    vs = client.vector_stores.create(name="Simple Vector Store")
    client.vector_stores.files.create_and_poll(vector_store_id=vs.id, file_id=file_id)
    print(f"[✓] Created vector store → ID: {vs.id}")
    return vs




# Going into the vectore store in order to query it further
def search_vector_store(vs, user_input):
    results = client.vector_stores.search(
        vector_store_id=vs.id,
        query=user_input, # Query line to search for vector
        rewrite_query=True,
        max_num_results=10
    )
    return results
 # Searching the vector store with the user input question



# Method to extract and place together text content from results that have a score above 0.5
def getobj(results):
    objlist = []
    for i in results.data:
        if i.score > 0.5:
            objlist.append("\n".join([chunk.text for chunk in i.content]))

    return objlist


# Taking out an answer from the model based on whats been produced
def synthesize_answer(chat_history, context_text, user_input):
    messages = [{"role": "system", "content": "You are a helpful legal assistant and your target audience is the general public that do not know the law, as well as barristers and also solicitors who are professionals in the law. Use the provided legal document in context to answer the user's question. If the act cannot be cited/utilised to answer the user's query, please state that this act is NOT RELEVANT by specifically stating that the act they provided does not apply to their question, but suggest a relevant act. When producing an answer for the user, please cite relevant/important sections. Additionally, if the Act doesn't appear to be relevant to their query and you cannot cite any sections, please state that the act is not relevant to their query, but do still answer their question about the law and suggest a relevant act the user can use instead."}] # Content of the system message - can be modified to change how the model will answer
    messages.extend(chat_history[-MAX_HISTORY_LENGTH:])
    messages.append({"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_input}"})

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages
    )

    return response.choices[0].message.content






file_id = file_upload("OAPA.pdf") # Uploading the file to cloud and getting back the file ID (unique) - storing in variable - calling function
vs = create_vector_store(file_id) # Creating the vector store and passing in the file ID from earlier to retrieve the vector store ID - calling function


# MVP = Minimal Viable Product

chat_history = []


while True:
        # Querying the model with a loop - allowing for multiple questions to be asked
        user_input = input("\nQuery: ").strip() 
        if user_input.lower() in ["exit", "quit"]:
            break
        print("Searching...")
        results = search_vector_store(vs, user_input)
        context_text = "\n".join([text for text in getobj(results)])
        print("Thinking...")
        answer =  synthesize_answer(chat_history, context_text, user_input)
        print("\n🤖 Answer:\n", answer)

        
        chat_history.append({"role": "user", "content": user_input}) # Storing user's question in chat_history (memory) for future use
        chat_history.append({"role": "assistant", "content": answer}) # Storing model's response in chat_history (memory) for future use





