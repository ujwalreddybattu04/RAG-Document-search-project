import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()  # load GOOGLE_API_KEY

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Add it to your .env file!")

genai.configure(api_key=api_key)

print("Fetching models...\n")
models = genai.list_models()

for m in models:
    print(
        m.name,
        "| generateContent supported:", 
        ("generateContent" in m.supported_generation_methods)
    )
