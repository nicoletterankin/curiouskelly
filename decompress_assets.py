import brotli
import os

# Paths
base_path = "public/unity/kelly-v1/Build"
files = [
    "kelly-v1.data.br",
    "kelly-v1.framework.js.br",
    "kelly-v1.wasm.br"
]

print("✨ Starting Decompression of Unity Assets (Brotli -> Raw)")

for filename in files:
    br_path = os.path.join(base_path, filename)
    # Remove .br extension for output
    out_path = os.path.join(base_path, filename[:-3])
    
    if os.path.exists(out_path):
        print(f"⏭️  Skipping {filename} (Uncompressed version exists)")
        continue
        
    if not os.path.exists(br_path):
        print(f"❌ Error: {filename} not found!")
        continue
        
    print(f"📦 Decompressing {filename}...")
    try:
        with open(br_path, 'rb') as f_in:
            compressed_data = f_in.read()
            decompressed_data = brotli.decompress(compressed_data)
            
        with open(out_path, 'wb') as f_out:
            f_out.write(decompressed_data)
            
        print(f"✅ Success: Created {out_path}")
    except Exception as e:
        print(f"❌ Failed to decompress {filename}: {e}")

print("🎉 All assets ready for loading!")













