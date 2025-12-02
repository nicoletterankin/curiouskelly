from vertexai.preview.vision_models import Image
import inspect

print("Inspecting Image class:")
print(dir(Image))

try:
    img = Image(b"fakebytes")
    print("\nImage instance attributes:")
    print(img.__dict__)
    # Try to see if there is a to_dict or similar
except Exception as e:
    print(f"Error creating image: {e}")
