import brotli
import os

# Input file (the Unity file with the custom header)
input_path = "public/unity/kelly-v1/Build/kelly-v1.data"
# Output file (the raw data without the custom Unity Brotli wrapper)
output_path = "public/unity/kelly-v1/Build/kelly-v1.decompressed.data"

print(f"🚀 Analyzing {input_path}...")

with open(input_path, 'rb') as f:
    header = f.read(64)
    print(f"Header: {header}")
    
    # Unity Header seems to be variable length text ending with null or similar
    # "UnityWeb Compressed Content (brotli)"
    
    # Let's try to find the start of the actual Brotli stream.
    # Brotli streams don't have a fixed magic header like GZIP, but we can try skipping the text.
    
    # Heuristic: Skip to the first byte that is NOT text-like, or try offsets.
    f.seek(0)
    full_data = f.read()

# Try to decompress from various offsets to find the valid stream
print("🕵️ Hunting for Brotli stream start...")

for offset in range(30, 100): # The header is around 30-50 bytes
    try:
        candidate_data = full_data[offset:]
        decompressed = brotli.decompress(candidate_data)
        print(f"✅ FOUND IT! Offset: {offset}")
        print(f"📦 Decompressed Size: {len(decompressed) / 1024 / 1024:.2f} MB")
        
        with open(output_path, 'wb') as out_f:
            out_f.write(decompressed)
        print(f"💾 Saved raw data to {output_path}")
        exit(0)
    except:
        pass

print("❌ Could not find valid Brotli stream in first 100 bytes.")











































