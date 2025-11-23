import json
import os
import time
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Import Vertex AI
import vertexai
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
from google.api_core.exceptions import NotFound, ResourceExhausted

load_dotenv()

@dataclass
class InteractionOption:
    text: str
    response: str
    learning_value: str  # 'High', 'Medium', 'Low'

@dataclass
class AtomContent:
    script: str
    options: List[InteractionOption]
    asl_gloss: str

class PersonaGenerator:
    """
    Generates Atomic Shards for specific Archetypes + Phases.
    Strategically falls back across models to ensure output.
    """

    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or "gen-lang-client-0005524332"
        self.location = "us-central1"
        self.model = None
        self.safety_config = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        
        self._init_vertex()

    def _init_vertex(self):
        try:
            vertexai.init(project=self.project_id, location=self.location)
            print(f"✅ Vertex AI Initialized: {self.project_id} @ {self.location}")
        except Exception as e:
            print(f"❌ Vertex AI Init Failed: {e}")

    def _get_model(self):
        # Try models in order of preference/speed
        candidates = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro", "gemini-pro"]
        
        for name in candidates:
            try:
                model = GenerativeModel(name)
                # Lightweight test
                # model.generate_content("test") 
                # Skip test to save time/quota, trusting catch block in main loop
                return model, name
            except Exception:
                continue
        return None, None

    def generate_atom(self, topic: str, core_fact: str, archetype: str, phase: str) -> AtomContent:
        prompt = f"""
        ### ROLE: THE ATOMIC ARCHITECT
        You are writing a script for 'Curious Kelly', an AI guide.
        Topic: {topic}
        Core Fact: "{core_fact}"
        Archetype: {archetype}
        Phase: {phase}
        
        TASK:
        Create a "Binary Choice" interaction where the user must choose between two distinct viewpoints or paths.
        
        OUTPUT JSON:
        {{
            "script": "Kelly's lines [asl:GESTURE]",
            "asl_gloss": "ASL GLOSS",
            "options": [
                {{ "text": "Option A (Viewpoint 1)", "response": "Kelly's response to A", "learning_value": "High" }},
                {{ "text": "Option B (Viewpoint 2)", "response": "Kelly's response to B", "learning_value": "High" }}
            ]
        }}
        """

        # Attempt generation with retries/fallbacks
        candidates = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
        
        for model_name in candidates:
            try:
                model = GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"},
                    safety_settings=self.safety_config
                )
                
                clean_json = response.text.replace('```json', '').replace('```', '').strip()
                data = json.loads(clean_json)
                
                # Parse options
                cleaned_options = []
                for opt in data.get('options', []):
                    cleaned_opt = {
                        "text": opt.get("text", opt.get("option", "")),
                        "response": opt.get("response", ""),
                        "learning_value": opt.get("learning_value", "Medium")
                    }
                    cleaned_options.append(InteractionOption(**cleaned_opt))
                
                return AtomContent(
                    script=data.get('script', ''),
                    options=cleaned_options,
                    asl_gloss=data.get('asl_gloss', '')
                )

            except ResourceExhausted:
                print(f"⚠️ Quota hit for {model_name}. Sleeping 10s...")
                time.sleep(10)
                continue # Try next model or retry same? For now, try next.
            except NotFound:
                 # Model not found, try next
                 continue
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                continue

        # FALLBACK: If all AI fails, return dummy content to unblock pipeline
        print(f"⚠️ ALL MODELS FAILED for {topic}/{archetype}. Using Emergency Backup.")
        return AtomContent(
            script=f"I am taking a moment to process {topic}. (System: Generation Offline)",
            options=[
                InteractionOption("Retry", "We will try again later.", "Low"),
                InteractionOption("Continue", "Moving on.", "Low")
            ],
            asl_gloss="SYSTEM OFFLINE PLEASE WAIT"
        )
