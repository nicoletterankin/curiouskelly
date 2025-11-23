import os

# Rename the files to remove .br extension
base_path = "public/unity/kelly-v1/Build"
files = [
    ("kelly-v1.data.br", "kelly-v1.data"),
    # ("kelly-v1.framework.js.br", "kelly-v1.framework.js"), # Already done
    # ("kelly-v1.wasm.br", "kelly-v1.wasm") # Already done
]

print("🔄 Renaming files to bypass extension confusion...")

for src, dst in files:
    src_path = os.path.join(base_path, src)
    dst_path = os.path.join(base_path, dst)
    
    if os.path.exists(src_path):
        # Check if dst exists, if so remove it (it might be the failed empty one from before)
        if os.path.exists(dst_path):
            os.remove(dst_path)
            
        os.rename(src_path, dst_path)
        print(f"✅ Renamed {src} -> {dst}")
    else:
        print(f"⚠️ {src} not found (maybe already renamed?)")

print("Done.")






