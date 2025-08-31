import os
from openai import OpenAI
import openai
from dotenv import load_dotenv
import os
import json

# MVP = Minimal Viable Product



# Load environment variables from .env file
load_dotenv()

# Initialising OpenAI client with API key from environment variable (hidden for security)
client = OpenAI(
    api_key= os.getenv("APIKEY")
)

# Constants / Configuration
OPENAI_MODEL = "gpt-4o-mini"
MAX_HISTORY_LENGTH = 6
CACHE_FILE = "test.json" # JSON file to store file IDs and vector store IDs
CHAT_HISTORY_FILE = "chathist.json" # JSON file to store chat history



# Uploading a file to OpenAI for use

def file_upload(filename):
    with open(filename, "rb") as f:
        file = client.files.create(file=f, purpose="assistants")
        print(f"[✓] Uploaded {filename} → ID: {file.id}")
        return file.id
    





# Creating a vector store (RAG) and uploading the file to it
# def create_vector_store(file_id):
#     vs = client.vector_stores.create(name="Simple Vector Store")
#     client.vector_stores.files.create_and_poll(vector_store_id=vs.id, file_id=file_id)
#     print(f"[✓] Created vector store → ID: {vs.id}")
#     return vs




# Going into the vector store in order to query it further
def search_vector_store(vector_store_id, user_input):
    results = client.vector_stores.search(
        vector_store_id=vector_store_id,
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
    messages.extend(chat_history[-MAX_HISTORY_LENGTH:]) # Adding the last few messages from chat history to provide context
    messages.append({"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_input}"})

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages
    )

    return response.choices[0].message.content

# Memory/Cache of file IDs and vector store IDs
def load_json(json_file_path):
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as f:
            return json.load(f)
    return {}

# Saving the JSON file with updated context
def save_json(context, json_file_path):
    with open(json_file_path, "w") as f:
        json.dump(context, f)

# Saving the file ID and Filename/**pathway!!!** to the JSON cache dictionary + saving the JSON file
def save_id_cache(file_id, json_store, filename, vector_store_id=None):
    for entry in json_store:
        if vector_store_id and entry.get("vector_store_ID") == vector_store_id:
            entry["file_IDs"][filename] = file_id
            break
    
    save_json(json_store, CACHE_FILE)


def pull_vs(vsname, filename):
    # Variables with function calling
    accessj = load_json(CACHE_FILE) # Loading the JSON file to access previous file IDs and vector store IDs
    new = False
    file_ids = {} # Initialising an empty dictionary to store file IDs
    vector_store_id = ""



    # Checking if the vector store already exists in the JSON cache dictionary - therefore wont need to create a new one
    for item in accessj:
        if "vector_store_name" in item and item["vector_store_name"] == vsname:
            vector_store_id = item.get("vector_store_ID", "")
            file_ids = item.get("file_IDs", {}) # Getting the dictionary of file IDs from the JSON cache
            break

    if not vector_store_id: # If the vector store ID doesn't exist, create a new vector store
        vs = client.vector_stores.create(name=vsname) # Creating the vector store and storing the ID in a variable IF it doesn't already exist in the JSON cache dictionary
        vector_store_id = vs.id
        new = True 
        print(f"[✓] Created vector store → ID: {vs.id}")




    # Checking if the file has already been uploaded by looking in the JSON cache dictionary (via filename) 
    if filename in file_ids:
        print(f"[i] File already uploaded. Using cached ID: {file_ids[filename]}")
        file_id = file_ids[filename]
    else: # If not uploaded, upload the file and save the ID to the JSON cache dictionary
        file_id = file_upload(filename) # Uploading the file to cloud and getting back the file ID (unique) - storing in variable - calling function
        if not new:
            save_id_cache(file_id, accessj, filename, vector_store_id)

        client.vector_stores.files.create_and_poll(vector_store_id=vector_store_id, file_id=file_id) # Adding the file to the vector store
        print(f"[✓] Added file to vector store → ID: {vector_store_id}")


    # If the file is new, add the vector store details and file ID to the JSON cache dictionary and save the JSON file
    if new: 
        new_file_ids = {}
        new_file_ids[filename] = file_id

        accessj.append({
            "vector_store_name": vsname,
            "vector_store_ID": vector_store_id,
            "file_IDs": new_file_ids
        })

        save_json(accessj, CACHE_FILE)

    return vector_store_id

# Variables that control the flow of the program afterwards
vsname = "Simple Vector Store"
filename = "OAPA.pdf" # Should be replaced with the path to the file you want to upload (interchangable later........)

vector_store_id = pull_vs(vsname, filename) # Calling the function to add the file and create/access the vector store - getting back the vector store ID



chat_history_id = 'theid' # This can be changed to allow for multiple chat histories to be stored and accessed
chat_history_db = load_json(CHAT_HISTORY_FILE) # Loading the chat history JSON file
chat_history = chat_history_db.get(chat_history_id, []) # Getting the chat history for the specific ID, or initialising an empty list if it doesn't exist (annoying bug fixed)


while True:
        # Querying the model with a loop - allowing for multiple questions to be asked
        user_input = input("\nQuery: ").strip() 
        if user_input.lower() in ["exit", "quit"]:
            break
        print("Searching...")
        results = search_vector_store(vector_store_id, user_input)
        context_text = "\n".join([text for text in getobj(results)])
        print("Thinking...")
        answer =  synthesize_answer(chat_history, context_text, user_input)
        print("\n🤖 Answer:\n", answer)

        
        chat_history.append({"role": "user", "content": user_input}) # Storing user's question in chat_history (memory) for future use
        chat_history.append({"role": "assistant", "content": answer}) # Storing model's response in chat_history (memory) for future use
        save_json({chat_history_id: chat_history}, CHAT_HISTORY_FILE) # Saving the chat history to a JSON file for future use


