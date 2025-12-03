import os

file_path = "public/unity/kelly-v1/Build/kelly-v1.data.br"

if os.path.exists(file_path):
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            print(f"File Header (Hex): {header.hex()}")
            
            # Simple check for GZIP (1f 8b)
            if header.startswith(b'\x1f\x8b'):
                print("Looks like GZIP")
            else:
                print("Does not look like standard GZIP. Might be Brotli or Raw.")
                
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print("File not found.")




















