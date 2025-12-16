import brotli
import os

input_path = "public/unity/kelly-v1/Build/kelly-v1.data.br"
output_path = "public/unity/kelly-v1/Build/kelly-v1.data"

print(f"🚀 Attempting to decompress {input_path} (735MB)...")

try:
    with open(input_path, 'rb') as f_in:
        data = f_in.read()
        print("📥 File read into memory.")
        
        decompressed = brotli.decompress(data)
        print("📦 Decompression successful!")
        
        with open(output_path, 'wb') as f_out:
            f_out.write(decompressed)
            print(f"✅ Saved to {output_path}")

except brotli.error as e:
    print(f"❌ Brotli Error: {e}")
except Exception as e:
    print(f"❌ General Error: {e}")
































