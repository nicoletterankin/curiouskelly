import json
import os
from dataclasses import dataclass
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@dataclass
class TimeTravelerContext:
    birth_year: int
    age: int
    generation_name: str
    cultural_touchstones: list[str]

class TimeTravelerGenerator:
    """
    Generates 'Chrono-Context' for lessons.
    Calculates what metaphors resonate with a specific birth year.
    """

    def __init__(self, model_name: str = "gemini-1.5-pro-latest"):
        self.model = genai.GenerativeModel(model_name)

    def calculate_context(self, age: int) -> TimeTravelerContext:
        current_year = datetime.now().year
        birth_year = current_year - age
        
        # Simple generation mapping (could be expanded)
        if birth_year < 1965: gen = "Boomer"
        elif birth_year < 1981: gen = "Gen X"
        elif birth_year < 1997: gen = "Millennial"
        elif birth_year < 2013: gen = "Gen Z"
        else: gen = "Alpha"
        
        return TimeTravelerContext(
            birth_year=birth_year,
            age=age,
            generation_name=gen,
            cultural_touchstones=[] # To be filled by AI
        )

    def generate_metaphors(self, topic: str, context: TimeTravelerContext) -> dict:
        """
        Asks Gemini for age-appropriate metaphors based on birth year history.
        """
        
        prompt = f"""
        ### ROLE: THE CULTURAL HISTORIAN
        You are an expert in generational sociology.
        
        ### TASK
        I need metaphors to explain the topic "{topic}" to someone born in {context.birth_year} (Age {context.age}, {context.generation_name}).
        
        ### CONSTRAINTS
        - Identify 3 cultural touchstones from when they were 10-20 years old.
        - Create a metaphor for "{topic}" using one of these touchstones.
        - The metaphor must be INTUITIVE to them, not forced.
        
        ### OUTPUT FORMAT (JSON)
        {{
            "touchstones": ["Event A", "Object B", "Trend C"],
            "primary_metaphor": "The concept is like [Object B] because...",
            "hook_line": "Remember [Object B]? [Topic] works exactly the same way."
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            clean_json = response.text.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_json)
        except Exception as e:
            print(f"❌ TIME TRAVEL ERROR: {e}")
            return {}

# --- TEST HARNESS ---
if __name__ == "__main__":
    tt = TimeTravelerGenerator()
    
    # Test for a 42-year-old (Born ~1983)
    ctx = tt.calculate_context(42)
    print(f"--- TESTING AGE {ctx.age} ({ctx.birth_year}) ---")
    
    meta = tt.generate_metaphors("Cloud Computing", ctx)
    print(json.dumps(meta, indent=2))






