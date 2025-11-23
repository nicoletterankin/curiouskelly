
import os
from google.cloud import aiplatform
from vertexai.preview import generative_models

project_id = "gen-lang-client-0005524332"
location = "us-central1"

aiplatform.init(project=project_id, location=location)

print(f"Listing models for {project_id} in {location}...")

try:
    # method 1: explicit list from model garden (harder to query via sdk simply, trying generation on standard names)
    import vertexai
    from vertexai.generative_models import GenerativeModel
    
    vertexai.init(project=project_id, location="us-west1")
    print("Testing in us-west1...")
    try:
        model = GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("Test")
        print("✅ SUCCESS: gemini-1.5-flash works in us-west1!")
    except Exception as e:
        print(f"❌ FAILED us-west1: {e}")

    vertexai.init(project=project_id, location="us-central1") # reset

    
    candidates = ["gemini-1.5-flash-001", "gemini-1.5-flash", "gemini-1.5-pro-001", "gemini-1.5-pro", "gemini-1.0-pro"]
    
    for model_name in candidates:
        print(f"\nTesting {model_name}...")
        try:
            model = GenerativeModel(model_name)
            response = model.generate_content("Test")
            print(f"✅ SUCCESS: {model_name} works!")
            break
        except Exception as e:
            print(f"❌ FAILED: {model_name} - {e}")

except Exception as e:
    print(f"Fatal error: {e}")

