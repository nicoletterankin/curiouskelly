#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import cv2

try:
	import mediapipe as mp
	from mediapipe.solutions.face_mesh_connections import FACEMESH_LIPS
except Exception:
	print('Please install mediapipe: pip install mediapipe', file=sys.stderr)
	sys.exit(1)

BASE_INPUT = Path('demo_output/kelly_diffuse_neutral_8k.png')
OUT_MASK = Path('demo_output/facial_expressions/mask_mouth.png')

mp_face_mesh = mp.solutions.face_mesh


def unique_indices(conns):
	ids = set()
	for a, b in conns:
		ids.add(a); ids.add(b)
	return sorted(ids)


def main():
	if not BASE_INPUT.exists():
		print(f'Missing base: {BASE_INPUT}', file=sys.stderr)
		sys.exit(1)
	OUT_MASK.parent.mkdir(parents=True, exist_ok=True)

	img = cv2.imread(str(BASE_INPUT), cv2.IMREAD_COLOR)
	h, w = img.shape[:2]

	# Detect landmarks on downscaled image for speed
	scale = 1280.0 / max(h, w)
	ds = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img.copy()
	with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as fm:
		res = fm.process(cv2.cvtColor(ds, cv2.COLOR_BGR2RGB))
		if not res.multi_face_landmarks:
			print('No face detected for mask.', file=sys.stderr)
			sys.exit(1)
		lm = res.multi_face_landmarks[0].landmark
		inv = (1.0/scale) if scale < 1.0 else 1.0
		pts = np.array([[p.x*ds.shape[1]*inv, p.y*ds.shape[0]*inv] for p in lm], dtype=np.float32)

	lip_idx = unique_indices(FACEMESH_LIPS)
	poly = pts[lip_idx].astype(np.int32)

	mask = np.zeros((h, w), dtype=np.uint8)
	cv2.fillConvexPoly(mask, cv2.convexHull(poly), 255)
	# Dilate outward to include cheeks/chin
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(w*0.01)|1, int(h*0.01)|1))
	mask = cv2.dilate(mask, kernel, iterations=1)
	# Feather edges slightly
	mask = cv2.GaussianBlur(mask, (9, 9), 0)

	cv2.imwrite(str(OUT_MASK), mask)
	print(f'Wrote mask: {OUT_MASK}')


if __name__ == '__main__':
	main()


























