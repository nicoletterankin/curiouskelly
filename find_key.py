
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
if key:
    print(f"Found Key: {key[:5]}...{key[-5:]}")
else:
    print("Key NOT found.")











































