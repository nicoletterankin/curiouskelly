#!/usr/bin/env python3
import os
import sys
import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from skimage.transform import PiecewiseAffineTransform, warp

try:
	import mediapipe as mp
except ImportError:
	print("Missing deps. Install: pip install mediapipe opencv-python scikit-image numpy", file=sys.stderr)
	sys.exit(1)

# Try to import connection groups, but fall back gracefully if unavailable
try:
	from mediapipe.solutions.face_mesh_connections import (
		FACEMESH_LIPS as _FACEMESH_LIPS,
		FACEMESH_LEFT_EYE as _FACEMESH_LEFT_EYE,
		FACEMESH_RIGHT_EYE as _FACEMESH_RIGHT_EYE,
		FACEMESH_LEFT_EYEBROW as _FACEMESH_LEFT_EYEBROW,
		FACEMESH_RIGHT_EYEBROW as _FACEMESH_RIGHT_EYEBROW,
	)
except Exception:
	_FACEMESH_LIPS = []
	_FACEMESH_LEFT_EYE = []
	_FACEMESH_RIGHT_EYE = []
	_FACEMESH_LEFT_EYEBROW = []
	_FACEMESH_RIGHT_EYEBROW = []

BASE_DIFFUSE = Path('demo_output/kelly_diffuse_neutral_8k.png')
OUT_DIR = Path('demo_output/facial_expressions')
META_PATH = OUT_DIR / 'facial_expressions_metadata.json'

mp_face_mesh = mp.solutions.face_mesh

# Landmark indices helpers
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291


def _unique_indices(conns: List[Tuple[int, int]]) -> List[int]:
	ids = set()
	for a, b in conns:
		ids.add(a)
		ids.add(b)
	return sorted(ids)


def read_bgr(path: Path) -> np.ndarray:
	img = cv2.imread(str(path), cv2.IMREAD_COLOR)
	if img is None:
		raise FileNotFoundError(f"Could not read image: {path}")
	return img


def detect_landmarks(image_bgr: np.ndarray) -> np.ndarray:
	h, w = image_bgr.shape[:2]
	scale = 1280.0 / max(h, w)
	img = cv2.resize(image_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else image_bgr
	with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as fm:
		res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		if not res.multi_face_landmarks:
			raise RuntimeError('No face landmarks detected')
		lm = res.multi_face_landmarks[0].landmark
		inv = (1.0/scale) if scale < 1.0 else 1.0
		pts = np.array([[p.x*img.shape[1]*inv, p.y*img.shape[0]*inv] for p in lm], dtype=np.float32)
		return pts


def roi_from_indices(pts: np.ndarray, idxs: List[int], w: int, h: int, pad_px: int = 30) -> Tuple[int, int, int, int]:
	region = pts[idxs]
	xmin, ymin = region.min(axis=0)
	xmax, ymax = region.max(axis=0)
	xmin = max(0, int(xmin - pad_px))
	ymin = max(0, int(ymin - pad_px))
	xmax = min(w - 1, int(xmax + pad_px))
	ymax = min(h - 1, int(ymax + pad_px))
	return xmin, ymin, xmax, ymax


def warp_region(image_bgr: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, idxs: List[int]) -> np.ndarray:
	h, w = image_bgr.shape[:2]
	xmin, ymin, xmax, ymax = roi_from_indices(src_pts, idxs, w, h, pad_px=40)
	roi = image_bgr[ymin:ymax+1, xmin:xmax+1].copy()
	if roi.size == 0:
		return image_bgr
	roi_h, roi_w = roi.shape[:2]
	src_roi = src_pts[idxs] - np.array([xmin, ymin], dtype=np.float32)
	dst_roi = dst_pts[idxs] - np.array([xmin, ymin], dtype=np.float32)
	# Stabilize with grid
	rows, cols = 12, 12
	grid_y = np.linspace(0, roi_h-1, rows)
	grid_x = np.linspace(0, roi_w-1, cols)
	gx, gy = np.meshgrid(grid_x, grid_y)
	grid = np.column_stack([gx.flatten(), gy.flatten()]).astype(np.float32)
	src_all = np.vstack([grid, src_roi])
	dst_all = np.vstack([grid, dst_roi])
	T = PiecewiseAffineTransform()
	T.estimate(dst_all, src_all)  # map dst->src for skimage.warp
	roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) / 255.0
	warped = warp(roi_rgb, T, output_shape=(roi_h, roi_w), mode='edge')
	warped_bgr = cv2.cvtColor((warped*255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
	# Feather composite
	mask = np.ones((roi_h, roi_w), dtype=np.float32)
	k = max(9, int(min(roi_h, roi_w)*0.05) | 1)
	mask = cv2.GaussianBlur(mask, (k, k), 0)[..., None]
	out = image_bgr.copy()
	out[ymin:ymax+1, xmin:xmax+1] = (mask*warped_bgr + (1.0-mask)*roi).astype(np.uint8)
	return out

# Expression builders ---------------------------------------------------------

def mouth_smile(dst: np.ndarray, src: np.ndarray, intensity: float, lips_idx: List[int]) -> np.ndarray:
	lc = src[LEFT_MOUTH_CORNER]
	rc = src[RIGHT_MOUTH_CORNER]
	d = np.linalg.norm(rc - lc)
	up = d * (0.02 * intensity)
	out = dst.copy()
	# Raise corners
	out[LEFT_MOUTH_CORNER] = [lc[0] - d*0.01*intensity, lc[1] - up]
	out[RIGHT_MOUTH_CORNER] = [rc[0] + d*0.01*intensity, rc[1] - up]
	# Slight lip shaping
	center_y = (lc[1] + rc[1])*0.5
	for i in lips_idx:
		p = src[i]
		if p[1] < center_y:
			out[i] = [p[0], p[1] - up*0.35]
		else:
			out[i] = [p[0], p[1] + up*0.25]
	return out


def mouth_frown(dst: np.ndarray, src: np.ndarray, intensity: float, lips_idx: List[int]) -> np.ndarray:
	lc = src[LEFT_MOUTH_CORNER]
	rc = src[RIGHT_MOUTH_CORNER]
	d = np.linalg.norm(rc - lc)
	down = d * (0.02 * intensity)
	out = dst.copy()
	out[LEFT_MOUTH_CORNER] = [lc[0], lc[1] + down]
	out[RIGHT_MOUTH_CORNER] = [rc[0], rc[1] + down]
	center_y = (lc[1] + rc[1])*0.5
	for i in lips_idx:
		p = src[i]
		if p[1] < center_y:
			out[i] = [p[0], p[1] + down*0.25]
		else:
			out[i] = [p[0], p[1] + down*0.35]
	return out


def mouth_open(dst: np.ndarray, src: np.ndarray, intensity: float, lips_idx: List[int]) -> np.ndarray:
	lc = src[LEFT_MOUTH_CORNER]
	rc = src[RIGHT_MOUTH_CORNER]
	d = np.linalg.norm(rc - lc)
	gap = d * (0.025 * intensity)
	out = dst.copy()
	center_y = (lc[1] + rc[1])*0.5
	for i in lips_idx:
		p = src[i]
		if p[1] < center_y:
			out[i] = [p[0], p[1] - gap]
		else:
			out[i] = [p[0], p[1] + gap]
	return out


def brow_raise(dst: np.ndarray, src: np.ndarray, intensity: float, idxs: List[int]) -> np.ndarray:
	out = dst.copy()
	# Raise brows by fraction of eye distance
	for i in idxs:
		p = src[i]
		out[i] = [p[0], p[1] - 6.0*intensity]
	return out


def brow_furrow(dst: np.ndarray, src: np.ndarray, intensity: float, left_idx: List[int], right_idx: List[int]) -> np.ndarray:
	out = dst.copy()
	for i in left_idx:
		p = src[i]
		out[i] = [p[0] + 2.5*intensity, p[1] + 2.0*intensity]
	for i in right_idx:
		p = src[i]
		out[i] = [p[0] - 2.5*intensity, p[1] + 2.0*intensity]
	return out


def eyes_blink(dst: np.ndarray, src: np.ndarray, intensity: float, eye_idx: List[int]) -> np.ndarray:
	out = dst.copy()
	# Compress vertically
	miny = src[eye_idx][:,1].min(); maxy = src[eye_idx][:,1].max(); cy = 0.5*(miny+maxy)
	for i in eye_idx:
		p = src[i]
		out[i] = [p[0], cy + (p[1]-cy)*(1.0-0.65*intensity)]
	return out


def eyes_wide(dst: np.ndarray, src: np.ndarray, intensity: float, eye_idx: List[int]) -> np.ndarray:
	out = dst.copy()
	miny = src[eye_idx][:,1].min(); maxy = src[eye_idx][:,1].max(); cy = 0.5*(miny+maxy)
	for i in eye_idx:
		p = src[i]
		out[i] = [p[0], cy + (p[1]-cy)*(1.0+0.25*intensity)]
	return out


def generate_batch():
	if not BASE_DIFFUSE.exists():
		print(f"Base texture not found: {BASE_DIFFUSE}", file=sys.stderr)
		sys.exit(1)
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	img = read_bgr(BASE_DIFFUSE)
	src_pts = detect_landmarks(img)
	# Index sets with graceful fallbacks
	lips_idx = _unique_indices(_FACEMESH_LIPS) if _FACEMESH_LIPS else list(range(61-10, 61+10)) + list(range(291-10, 291+10))
	left_eye_idx = _unique_indices(_FACEMESH_LEFT_EYE) if _FACEMESH_LEFT_EYE else list(range(33-5, 33+5))
	right_eye_idx = _unique_indices(_FACEMESH_RIGHT_EYE) if _FACEMESH_RIGHT_EYE else list(range(263-5, 263+5))
	left_brow_idx = _unique_indices(_FACEMESH_LEFT_EYEBROW) if _FACEMESH_LEFT_EYEBROW else list(range(105-5, 105+5))
	right_brow_idx = _unique_indices(_FACEMESH_RIGHT_EYEBROW) if _FACEMESH_RIGHT_EYEBROW else list(range(334-5, 334+5))

	plans = [
		('happy', 'emotion', lambda d: mouth_smile(d, src_pts, 1.0, lips_idx), [lips_idx]),
		('sad', 'emotion', lambda d: mouth_frown(d, src_pts, 1.0, lips_idx), [lips_idx]),
		('mouth_open_small', 'mouth', lambda d: mouth_open(d, src_pts, 0.35, lips_idx), [lips_idx]),
		('mouth_open_medium', 'mouth', lambda d: mouth_open(d, src_pts, 0.65, lips_idx), [lips_idx]),
		('mouth_open_large', 'mouth', lambda d: mouth_open(d, src_pts, 1.0, lips_idx), [lips_idx]),
		('brow_raise_both', 'eyebrow', lambda d: brow_raise(d, src_pts, 1.0, left_brow_idx+right_brow_idx), [left_brow_idx+right_brow_idx]),
		('brow_furrow_both', 'eyebrow', lambda d: brow_furrow(d, src_pts, 1.0, left_brow_idx, right_brow_idx), [left_brow_idx, right_brow_idx]),
		('eye_blink_left', 'eye', lambda d: eyes_blink(d, src_pts, 1.0, left_eye_idx), [left_eye_idx]),
		('eye_blink_right', 'eye', lambda d: eyes_blink(d, src_pts, 1.0, right_eye_idx), [right_eye_idx]),
		('eye_blink_both', 'eye', lambda d: eyes_blink(eyes_blink(d, src_pts, 1.0, left_eye_idx), src_pts, 1.0, right_eye_idx), [left_eye_idx, right_eye_idx]),
		('eye_wide_both', 'eye', lambda d: eyes_wide(eyes_wide(d, src_pts, 1.0, left_eye_idx), src_pts, 1.0, right_eye_idx), [left_eye_idx, right_eye_idx]),
	]

	metadata: Dict[str, Dict[str, str]] = {}
	for name, typ, build_fn, regions in plans:
		dst_pts = src_pts.copy()
		dst_pts = build_fn(dst_pts)
		out_img = img.copy()
		for region in regions:
			out_img = warp_region(out_img, src_pts, dst_pts, region)
		out_path = OUT_DIR / f'kelly_{name}_8k.png'
		cv2.imwrite(str(out_path), out_img)
		metadata[name] = {
			'path': str(out_path).replace('\\', '/'),
			'type': typ,
			'description': f'Auto-generated {name} using landmark warp'
		}
		print(f"Generated: {out_path}")

	with open(META_PATH, 'w', encoding='utf-8') as f:
		json.dump(metadata, f, indent=2)
	print(f"Wrote metadata: {META_PATH}")


def main():
	try:
		generate_batch()
	except Exception as e:
		print(f"ERROR: {e}", file=sys.stderr)
		sys.exit(1)


if __name__ == '__main__':
	main()
