#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import urllib.request
import replicate

BASE_INPUT = Path('demo_output/kelly_diffuse_neutral_8k.png')
MASK_PATH = Path('demo_output/facial_expressions/mask_mouth.png')
OUT_DIR = Path('demo_output/facial_expressions')
OUT_NAME = 'kelly_happy_inpaint_8k.png'

# Replicate SDXL inpainting model (common community fork)
MODEL = 'stability-ai/sdxl-inpainting'
VERSION = '1f14a2c8900e3f2f1f579b0d7f42d1f1b4b6f3d8e1f7f5c7f5a66b1db8f08a4b'  # may change; replicate client accepts model:version


def main():
	if 'REPLICATE_API_TOKEN' not in os.environ:
		print('ERROR: Set REPLICATE_API_TOKEN in environment to use this script.', file=sys.stderr)
		sys.exit(1)
	if not BASE_INPUT.exists():
		print(f'ERROR: Missing base input image: {BASE_INPUT}', file=sys.stderr)
		sys.exit(1)
	if not MASK_PATH.exists():
		print(f'ERROR: Missing mask image: {MASK_PATH}. Run scripts/make_mouth_mask.py first.', file=sys.stderr)
		sys.exit(1)

	OUT_DIR.mkdir(parents=True, exist_ok=True)

	client = replicate.Client(api_token=os.environ['REPLICATE_API_TOKEN'])

	prompt = (
		"Photorealistic inpainting. Preserve original identity, exact pose, framing, clothing, and background. "
		"Edit ONLY the masked mouth/cheeks region to a subtle genuine smile with natural teeth visibility, "
		"no other changes, no extra people, no composition changes, realistic skin texture."
	)

	with open(BASE_INPUT, 'rb') as image_f, open(MASK_PATH, 'rb') as mask_f:
		inputs = {
			'image': image_f,
			'mask': mask_f,
			'prompt': prompt,
			'negative_prompt': 'second person, extra person, body change, new background, deformation, cartoon',
			'num_inference_steps': 40,
			'guidance_scale': 6.5,
			'width': 1024,
			'height': 1024,
			'seed': 12345,
		}
		print('Submitting SDXL inpainting...')
		result = client.run(f"{MODEL}:{VERSION}", input=inputs)

	if not result:
		print('ERROR: Empty result from model', file=sys.stderr)
		sys.exit(1)

	url = result[0] if isinstance(result, list) else str(result)
	out_path = OUT_DIR / OUT_NAME
	print(f'Downloading result to {out_path}...')
	urllib.request.urlretrieve(url, out_path)
	print(f'Saved: {out_path}')


if __name__ == '__main__':
	try:
		main()
	except Exception as e:
		print(f'ERROR: {e}', file=sys.stderr)
		sys.exit(1)



























