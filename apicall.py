import os
from openai import OpenAI
import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()





client = OpenAI(
    api_key= os.getenv("APIKEY")
)

# response = client.responses.create(
#     model="gpt-4.1-nano",
#     instructions="You are a coding assistant that talks like a pirate.",
#     input="How do I check if a Python object is an instance of a class?",
# )

#print(response.output_text)

# with open("weightandmeasuresact.pdf", "r") as f:
#             f = f.read()
