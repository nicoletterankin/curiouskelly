import os

env_path = ".env"

try:
    with open(env_path, "rb") as f:
        content = f.read()
    
    # Check for UTF-16 BOM (FF FE) or UTF-8 BOM (EF BB BF)
    if content.startswith(b'\xff\xfe'):
        print("Found UTF-16 LE BOM. Fixing...")
        content = content[2:].decode('utf-16').encode('utf-8')
    elif content.startswith(b'\xfe\xff'):
        print("Found UTF-16 BE BOM. Fixing...")
        content = content[2:].decode('utf-16-be').encode('utf-8')
    elif content.startswith(b'\xef\xbb\xbf'):
        print("Found UTF-8 BOM. Stripping...")
        content = content[3:]
    else:
        print("No BOM found. File might be clean or using another encoding.")
        
    # Write back as clean UTF-8
    with open(env_path, "wb") as f:
        f.write(content)
        
    print("✅ .env file normalized to UTF-8")

except Exception as e:
    print(f"❌ Error fixing file: {e}")






