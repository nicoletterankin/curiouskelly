
import os
import sys
from dotenv import load_dotenv

# Load env vars
load_dotenv()

project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or "gen-lang-client-0005524332"

print(f"Checking Vertex AI access for project: {project_id}")

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    
    vertexai.init(project=project_id, location="us-central1")
    
    model = GenerativeModel("gemini-1.5-flash")
    print("Attempting to generate text via Vertex AI (gemini-1.5-flash)...")
    
    response = model.generate_content("Say 'Hello from Vertex AI' if you can hear me.")
    
    print("\nSUCCESS! Generated text:")
    print(response.text)
    print("\nThis confirms your Paid/GCP account is accessible via Vertex AI.")

except Exception as e:
    print(f"\nFAILURE: Could not generate with Vertex AI.")
    print(f"Error: {e}")
    print("\nTroubleshooting:")
    print("1. Ensure 'gcloud auth application-default login' has been run.")
    print("2. Ensure the project ID is correct.")
    print("3. Ensure the Vertex AI API is enabled in Google Cloud Console.")

