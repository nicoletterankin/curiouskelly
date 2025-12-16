# 🚨 URGENT FIX REQUIRED: Database Schema Missing

## Problem
Your `generate_all_atoms.py` script is crashing with:
```
CRITICAL_ERROR: relation "core_lessons" does not exist
```

This means the database tables were never created in your Supabase instance.

## Root Cause
You deployed schema to one Supabase project, but your `.env` is pointing to a different one.

---

## Fix Instructions (Execute Now)

### Step 1: Verify Database Connection
Run this diagnostic script:

**File: `src/scripts/check_db.py`**
```python
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv(override=True)

DATABASE_URL = os.getenv('DATABASE_URL')
print(f"🔍 Connecting to: {DATABASE_URL[:50]}...")

try:
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    # Check if tables exist
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """)
    
    tables = cursor.fetchall()
    print(f"\n✅ Connected! Found {len(tables)} tables:")
    for table in tables:
        print(f"  - {table[0]}")
    
    if not tables:
        print("\n❌ NO TABLES FOUND! You need to deploy the schema.")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"\n❌ Connection failed: {e}")
```

**Run:**
```bash
cd C:\Users\user\UI-TARS-desktop\curious-kellly\content-engine
python src/scripts/check_db.py
```

---

### Step 2: Deploy Schema (If Missing)
If Step 1 shows no tables, run:

```bash
python src/scripts/deploy_schema.py
```

This will create:
- `core_lessons` table
- `lesson_atoms` table

---

### Step 3: Insert Core Lessons (If Empty)
Check if `core_lessons` is empty:

```bash
python -c "import psycopg2, os; from dotenv import load_dotenv; load_dotenv(override=True); conn = psycopg2.connect(os.getenv('DATABASE_URL')); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM core_lessons'); print(f'Core Lessons: {cursor.fetchone()[0]}'); conn.close()"
```

If it returns `0`, run:

```bash
python src/scripts/bulk_insert_core.py
```

---

### Step 4: Restart Atom Generation
Once database has:
- ✅ Tables created
- ✅ 365 core lessons inserted

Restart the generation:

```bash
python src/scripts/generate_all_atoms.py
```

---

## Verification Checklist
Before restarting generation, confirm:

- [ ] `check_db.py` shows `core_lessons` and `lesson_atoms` tables exist
- [ ] `core_lessons` table has 365 rows
- [ ] `.env` file has correct `DATABASE_URL` (no typos, correct password)
- [ ] No other `.env` files in parent directories overriding credentials

---

## Prevention
Your `generate_all_atoms.py` should check for tables before starting. Add this to the top of the script:

```python
def validate_database():
    cursor.execute("""
        SELECT COUNT(*) FROM core_lessons
    """)
    count = cursor.fetchone()[0]
    
    if count == 0:
        raise Exception("❌ Database is empty! Run bulk_insert_core.py first.")
    
    print(f"✅ Database ready: {count} core lessons found.")
```

---

## Expected Output After Fix
```
✅ Database ready: 365 core lessons found.
Generating atoms: 0/21900 [00:00<?, ?it/s]
```

---

## If Still Failing
1. Share the output of `check_db.py`
2. Share the first 100 characters of your `DATABASE_URL` (redact password)
3. Confirm which Supabase project you're using (the dashboard URL)
































