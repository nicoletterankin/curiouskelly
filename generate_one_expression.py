#!/usr/bin/env python3
import os
from PIL import Image, ImageDraw
import numpy as np

BASE_DIFFUSE = os.path.join('demo_output', 'kelly_diffuse_neutral_8k.png')
OUT_DIR = os.path.join('demo_output', 'facial_expressions')
OUT_PATH = os.path.join(OUT_DIR, 'kelly_happy_8k.png')


def ensure_dirs():
	os.makedirs(OUT_DIR, exist_ok=True)


def generate_happy_from_diffuse():
	if not os.path.exists(BASE_DIFFUSE):
		raise FileNotFoundError(f"Base diffuse not found: {BASE_DIFFUSE}")

	img = Image.open(BASE_DIFFUSE).convert('RGB')
	draw = ImageDraw.Draw(img)

	# Approximate mouth region for a 8K portrait (rough placeholder)
	mouth = (3000, 2800, 5000, 3200)
	center_x = (mouth[0] + mouth[2]) // 2
	center_y = (mouth[1] + mouth[3]) // 2
	width = int((mouth[2] - mouth[0]) * 0.8)
	height = int((mouth[3] - mouth[1]) * 0.3)

	points = []
	for i in range(0, width, 20):
		x = center_x - width // 2 + i
		y = center_y + int(height * np.sin(i * np.pi / width))
		points.append((x, y))
	if len(points) > 1:
		draw.line(points, fill=(0, 0, 0), width=4)

	img.save(OUT_PATH)
	return OUT_PATH


def main():
	ensure_dirs()
	out = generate_happy_from_diffuse()
	print(f"Generated: {out}")


if __name__ == '__main__':
	main()


























