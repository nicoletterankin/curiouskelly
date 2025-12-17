import brotli
import os

# Fallback: It might NOT be Brotli. It might be LZ4 or just raw with a header.
# But the header said "brotli".
# It is possible the file is corrupted or using a custom Unity dictionary.

# Plan C: The "WebAssembly" approach.
# We can't decompress it here. We must rely on the browser.
# But the browser fails because of the header.
# We will STRIP the header and serve it as .br to see if the browser accepts it then.

input_path = "public/unity/kelly-v1/Build/kelly-v1.data"
output_path = "public/unity/kelly-v1/Build/kelly-v1.stripped.data.br"

with open(input_path, 'rb') as f:
    data = f.read()

# The header is: k \x8d \x00 UnityWeb Compressed Content (brotli)
# Length: 38 chars for text + 3 bytes prefix = 41 bytes?
# Text: UnityWeb Compressed Content (brotli) = 38 bytes
# Prefix: 6b 8d 00 = 3 bytes.
# Total = 41 bytes.

# Let's try stripping the first 41 bytes and see if that makes it valid brotli.
# Or 42, 43, 44...

print("✂️  Creating stripped versions for browser testing...")

# Version A: Strip 41 bytes
with open(output_path, 'wb') as f:
    f.write(data[41:])
print(f"Created {output_path} (Offset 41)")

# Version B: Strip variable
# We will just rely on the "Hybrid" server again but this time we know
# we can't decompress it.



































