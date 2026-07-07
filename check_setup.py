import os
from dotenv import load_dotenv
import openai
import pydantic

# Load variables from .env into the environment
load_dotenv()

# Read the API key
api_key = os.getenv("OPENAI_API_KEY")

print("=" * 50)
print("ENVIRONMENT CHECK")
print("=" * 50)
print(f"OpenAI library version : {openai.__version__}")
print(f"Pydantic library version: {pydantic.__version__}")

if api_key and api_key.startswith("sk-"):
    print("OPENAI_API_KEY loaded  : ✅ Found (starts with sk-...)")
else:
    print("OPENAI_API_KEY loaded  : ❌ NOT FOUND or invalid")

print("=" * 50)