# 🚨 URGENT: Generation Script Crashed - Fix Required

## Problems Detected

### Issue 1: Duplicate Key Constraint
**Error:** `duplicate key value violates unique constraint "lesson_atoms_core_lesson_id_archetype_phase_key"`

**Cause:** Script is trying to insert atoms that already exist (193 atoms from previous run)

**Fix:** Add "skip if exists" logic to the generation script

---

### Issue 2: Gemini JSON Malformed
**Error:** `InteractionOption.__init__() got an unexpected keyword argument 'asl_gloss'`

**Cause:** Gemini is placing `asl_gloss` at the wrong level in the JSON structure

**Fix:** Update the prompt to be more explicit about JSON structure, and add JSON validation/cleanup

---

## Fix Script for `generate_all_atoms.py`

### Step 1: Add Skip-if-Exists Logic

**File:** `src/scripts/generate_all_atoms.py`

Add this function at the top:

```python
def atom_exists(cursor, lesson_id, archetype, phase):
    """Check if atom already exists"""
    cursor.execute("""
        SELECT COUNT(*) FROM lesson_atoms 
        WHERE core_lesson_id = %s 
        AND archetype = %s 
        AND phase = %s
    """, (lesson_id, archetype, phase))
    return cursor.fetchone()[0] > 0
```

Then modify the generation loop to:

```python
for phase in ['Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom']:
    # CHECK IF EXISTS FIRST
    if atom_exists(cursor, lesson_id, archetype_name, phase):
        print(f"⏭️  {topic}/{archetype_name}/{phase}: Already exists, skipping")
        continue
    
    # Generate atom
    try:
        atom = generator.generate_atom(...)
        # ... save to DB
    except Exception as e:
        print(f"❌ {topic}/{archetype_name}/{phase}: {e}")
        continue  # Don't crash, just log and move on
```

---

### Step 2: Fix ASL Gloss in Prompt

**File:** `src/generators/persona_generator.py`

Update the prompt to be VERY explicit about JSON structure:

```python
prompt = f"""
Generate an Atomic Shard for Curious Kelly.

TOPIC: {topic}
UNIVERSAL TRUTH: {universal_truth}
ARCHETYPE: {archetype_name}
PHASE: {phase}

OUTPUT MUST BE VALID JSON IN THIS EXACT FORMAT:
{{
  "script": "What Kelly says (include [asl:WAVE] markers inline)",
  "options": [
    {{"text": "Option 1"}},
    {{"text": "Option 2"}},
    {{"text": "Option 3"}}
  ],
  "responses": {{
    "Option 1": "Response to option 1",
    "Option 2": "Response to option 2",
    "Option 3": "Response to option 3"
  }}
}}

CRITICAL RULES:
1. ASL markers go INSIDE the "script" text as [asl:GESTURE]
2. NO "asl_gloss" field in options
3. Options array contains objects with ONLY "text" field
4. Return ONLY the JSON, no markdown, no explanation
"""
```

---

### Step 3: Add JSON Validation

Add this function to clean Gemini's response:

```python
import json
import re

def clean_gemini_json(text):
    """Extract and clean JSON from Gemini response"""
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Parse JSON
    data = json.loads(text.strip())
    
    # Clean options: remove any asl_gloss keys
    if 'options' in data:
        cleaned_options = []
        for opt in data['options']:
            if isinstance(opt, dict):
                cleaned_options.append({"text": opt.get("text", opt.get("option", ""))})
            else:
                cleaned_options.append({"text": str(opt)})
        data['options'] = cleaned_options
    
    return data
```

Then use it:

```python
response = model.generate_content(prompt)
cleaned_json = clean_gemini_json(response.text)
atom = InteractionAtom(**cleaned_json)
```

---

## Quick Fix Commands

Run these in order:

### 1. Stop current broken generation
```bash
# Find the process
Get-Process python | Where-Object {$_.Id -eq 64660}

# Kill it (if still running)
Stop-Process -Id 64660
```

### 2. Check what's in the database
```bash
cd curious-kellly/content-engine
python scripts/status.py
```

### 3. Apply fixes to the code
Edit the files above with the fixes

### 4. Restart generation
```bash
python scripts/generate_all_atoms.py
```

---

## Verification

After restarting, you should see:

```
⏭️  Leaves/The Survivor/Hook: Already exists, skipping
⏭️  Leaves/The Survivor/Fact1: Already exists, skipping
...
✅ Leaves/The Rebel/Hook: Generated successfully
```

---

## Prevention

Add to the top of `generate_all_atoms.py`:

```python
# Pre-flight check
cursor.execute("SELECT COUNT(*) FROM lesson_atoms")
existing = cursor.fetchone()[0]
print(f"🔍 Found {existing} existing atoms, will skip these")
```

This way you'll know if atoms already exist before starting.



















