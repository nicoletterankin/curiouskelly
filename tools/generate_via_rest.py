import argparse
import base64
import json
import os
import requests
import google.auth
from google.auth.transport.requests import Request

def get_credentials():
    credentials, project_id = google.auth.default()
    credentials.refresh(Request())
    return credentials, project_id

def generate_image(prompt, output_file, reference_image_path=None, project_id=None):
    credentials, default_project_id = get_credentials()
    project_id = project_id or default_project_id
    
    if not project_id:
        raise ValueError("Project ID not found")

    print(f"Using project: {project_id}")
    
    token = credentials.token
    location = "us-central1"
    model = "imagegeneration@006"
    
    url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model}:predict"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }
    
    # Append [1] to prompt if reference is used
    final_prompt = prompt
    if reference_image_path:
        final_prompt = f"{prompt} [1]"

    instance = {
        "prompt": final_prompt
    }
    
    if reference_image_path and os.path.exists(reference_image_path):
        with open(reference_image_path, "rb") as f:
            image_content = f.read()
            encoded_image = base64.b64encode(image_content).decode("utf-8")
            
        instance["referenceImages"] = [
            {
                "referenceType": "REFERENCE_TYPE_SUBJECT",
                "referenceId": 1,
                "referenceImage": {
                    "bytes": encoded_image
                },
                "subjectImageConfig": {
                    "subjectDescription": "Kelly Rein, photorealistic digital human, oval face with soft rounded contours, long hair extending well past shoulders",
                    "subjectType": "SUBJECT_TYPE_PERSON"
                }
            }
        ]
        print(f"Attaching reference image: {reference_image_path} (ID: 1)")

    payload = {
        "instances": [instance],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": "16:9",
            "personGeneration": "allow_adult"
        }
    }
    
    print("Sending request...")
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    response_json = response.json()
    
    try:
        predictions = response_json.get("predictions", [])
        if not predictions:
            print("No predictions returned.")
            print(response_json)
            return

        bytes_base64 = predictions[0].get("bytesBase64Encoded")
        if bytes_base64:
            with open(output_file, "wb") as f:
                f.write(base64.b64decode(bytes_base64))
            print(f"✅ Saved to {output_file}")
        else:
            print("No image bytes in response.")
            print(response_json)

    except Exception as e:
        print(f"Error processing response: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", help="Path to reference image")
    parser.add_argument("--project", help="Project ID")
    args = parser.parse_args()
    
    generate_image(args.prompt, args.output, args.reference, args.project)
