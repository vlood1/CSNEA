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

user_input = "Has anyone been sent to prison via this act?" # Example user input question


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

messages = [{"role": "system", "content": "You are a helpful legal assistant and your target audience is the general public that do not know the law, as well as barristers and also solicitors who are professionals in the law. Use the provided legal document in context to answer the user's question. If the act cannot be cited/utilised to answer the user's query, please state that this act is NOT RELEVANT by specifically stating that the act they provided does not apply to their question, but suggest a relevant act. When producing an answer for the user, please cite relevant/important sections. Additionally, if the Act doesn't appear to be relevant to their query and you cannot cite any sections, please state that the act is not relevant to their query, but do still answer their question about the law and suggest a relevant act the user can use instead."}] # Content of the system message - can be modified to change how the model will answer
messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"})


# Saving LLM's answer (memory) - assistant
messages.append({"role": "assistant", "content": '''Under the Offences Against the Person Act 1861, if you are convicted of murder, the punishment is severe. Specifically:

1. **Death Penalty**: Whosoever shall be convicted of Murder shall suffer Death as a Felon (Section 1).
2. **Sentence for Murder**: Upon every conviction for murder, the Court shall pronounce a sentence of death (Section 2).

It's important to note, though, that the death penalty in the UK has been abolished; thus, while historically, a conviction for murder would have carried the death penalty, current legal systems provide for life imprisonment or other serious penalties for such a crime.

For the historical context as provided:
- In the Act, it states that conviction leads to execution (Section 2), but this applies to the law as it was originally enacted and does not reflect current legal practices, which have evolved significantly since then.

Therefore, if you were convicted of murder today, you would likely face life imprisonment or a lengthy prison sentence instead.'''})




messages.append({"role": "user", "content": "How's your day?"})




# Response creation
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
    )

# ADD WAY TO REMEMBER THINGS! (messages list) ^^^

print(response.choices[0].message.content)



