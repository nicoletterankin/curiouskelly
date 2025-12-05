import brotli
import os

input_path = "public/unity/kelly-v1/Build/kelly-v1.data"
output_path = "public/unity/kelly-v1/Build/kelly-v1.decompressed.data"

print(f"🚀 Analyzing {input_path}...")

with open(input_path, 'rb') as f:
    full_data = f.read()

# The header we saw was: b'k\x8d\x00UnityWeb Compressed Content (brotli)'
# The Brotli stream likely starts immediately after this string or close to it.
# The header bytes printed were:
# 6b 8d 00 55 6e 69 74 79 57 65 62 20 43 6f 6d 70 72 65 73 73 65 64 20 43 6f 6e 74 65 6e 74 20 28 62 72 6f 74 6c 69 29
# 'k' .. .. U n i t y W e b ... (brotli)

# Length of "UnityWeb Compressed Content (brotli)" is 38 bytes.
# Plus the first 3 bytes (k \x8d \x00) = 41 bytes?

# Let's just try every byte from 0 to 200. The previous script might have errored silently or missed it.
print("🕵️ Deep Scan for Brotli stream start...")

success = False
for offset in range(0, 200):
    try:
        candidate_data = full_data[offset:]
        # Try decompressing just the first chunk to be faster
        brotli.decompress(candidate_data[:1024]) 
        
        # If that worked, do the whole thing
        print(f"✅ FOUND IT! Offset: {offset}")
        decompressed = brotli.decompress(candidate_data)
        print(f"📦 Decompressed Size: {len(decompressed) / 1024 / 1024:.2f} MB")
        
        with open(output_path, 'wb') as out_f:
            out_f.write(decompressed)
        print(f"💾 Saved raw data to {output_path}")
        success = True
        break
    except Exception as e:
        # print(f"Offset {offset} failed: {e}")
        pass

if not success:
    print("❌ Critical Failure: Could not find valid Brotli stream.")





















