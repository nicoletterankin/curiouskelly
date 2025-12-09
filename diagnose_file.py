import gzip
import zlib
import struct

file_path = "public/unity/kelly-v1/Build/kelly-v1.data.br"

def get_head(n=32):
    with open(file_path, 'rb') as f:
        return f.read(n)

header = get_head()
print(f"Header Hex: {header.hex()}")
print(f"Header ASCII: {header}")

# Test GZIP
try:
    with gzip.open(file_path, 'rb') as f:
        data = f.read(100)
        print("✅ GZIP Decompression: SUCCESS")
except Exception as e:
    print(f"❌ GZIP Decompression: FAILED ({e})")

# Test ZLIB
try:
    with open(file_path, 'rb') as f:
        data = zlib.decompress(f.read(1000))
        print("✅ ZLIB Decompression: SUCCESS")
except Exception as e:
    print(f"❌ ZLIB Decompression: FAILED ({e})")

# Test if it's actually uncompressed Unity Data (look for UnityWebData)
if b"UnityWebData" in header:
    print("✅ Format: Raw Unity Web Data (Uncompressed)")
else:
    print("ℹ️  Format: Not explicit UnityWebData signature")






















