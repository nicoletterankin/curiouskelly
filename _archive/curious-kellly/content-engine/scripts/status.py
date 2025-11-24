"""Quick status checker for atom generation"""
import os
import sys
from dotenv import load_dotenv
import psycopg2
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path, override=True)

DATABASE_URL = os.getenv('DATABASE_URL')

try:
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    # Get counts
    cursor.execute("SELECT COUNT(*) FROM core_lessons")
    lessons = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM lesson_atoms")
    atoms = cursor.fetchone()[0]
    
    # Calculate progress
    target = 21900
    progress = (atoms / target) * 100
    remaining = target - atoms
    
    # Estimate time (3.6 sec per atom)
    seconds_left = remaining * 3.6
    hours_left = seconds_left / 3600
    eta = datetime.now() + timedelta(seconds=seconds_left)
    
    print("=" * 60)
    print("🚀 ANTIGRAVITY FACTORY STATUS")
    print("=" * 60)
    print(f"📚 Core Lessons:     {lessons}/365")
    print(f"⚛️  Atoms Generated:  {atoms:,}/{target:,}")
    print(f"📊 Progress:         {progress:.2f}%")
    print(f"⏱️  Remaining:        {remaining:,} atoms")
    print(f"🎯 ETA:              ~{hours_left:.1f} hours ({eta.strftime('%I:%M %p %b %d')})")
    print("=" * 60)
    
    if atoms < 100:
        print("⚠️  Generation may be slow at startup (warming up API)")
    elif progress > 95:
        print("🎉 Almost done! Stay tuned for completion!")
    else:
        print("✅ Generation running smoothly")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)






