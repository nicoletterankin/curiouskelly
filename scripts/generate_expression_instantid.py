#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import urllib.request
import replicate

BASE_INPUT = Path('demo_output/kelly_chair_diffuse_neutral_8k.png')
OUT_DIR = Path('demo_output/facial_expressions')
OUT_NAME = 'kelly_happy_chair_8k.png'

# Replicate model slug and version (update if needed)
MODEL = 'zsxkib/instant-id'
VERSION = '491ddf5be6b827f8931f088ef10c6d015f6d99685e6454e6f04c8ac298979686'


def main():
	if 'REPLICATE_API_TOKEN' not in os.environ:
		print('ERROR: Set REPLICATE_API_TOKEN in environment to use this script.', file=sys.stderr)
		sys.exit(1)
	if not BASE_INPUT.exists():
		print(f'ERROR: Missing base input image: {BASE_INPUT}', file=sys.stderr)
		sys.exit(1)

	OUT_DIR.mkdir(parents=True, exist_ok=True)

	client = replicate.Client(api_token=os.environ['REPLICATE_API_TOKEN'])

	prompt = (
		"Preserve the exact person (Kelly), SAME pose, SAME framing, SAME clothing (gray sweater), and SAME background with chair. "
		"Apply a subtle genuine smile: mouth corners slightly raised, cheeks gently lifted, eyes engaged. "
		"No other changes."
	)
	negative = (
		"second person, extra person, added body, different pose, arm pose change, dress, wedding dress, gown, new background, "
		"hands on face, hand near face, camera change, low quality, blur, cartoon, deformed"
	)

	# Pass a file handle directly; the client uploads it for us
	with open(BASE_INPUT, 'rb') as f:
		inputs = {
			'image': f,
			'prompt': prompt,
			'negative_prompt': negative,
			'cfg': 4.0,
			'steps': 35,
			'seed': 4321,
			'instantid_weight': 0.92,
			'width': 1536,
			'height': 864,
		}
		print('Submitting to Replicate InstantID...')
		result = client.run(f"{MODEL}:{VERSION}", input=inputs)

	if not result:
		print(f'ERROR: Empty result from model', file=sys.stderr)
		sys.exit(1)

	# Expect a list of URLs
	if isinstance(result, list):
		url = result[0]
	else:
		url = str(result)

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
