#!/usr/bin/env python3
import os
import sys
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple

# Free local dependencies
import cv2
from skimage.transform import PiecewiseAffineTransform, warp

try:
	import mediapipe as mp
except ImportError as e:
	print("mediapipe not installed. Please run: pip install mediapipe opencv-python scikit-image numpy", file=sys.stderr)
	raise

BASE_DIFFUSE = Path('demo_output/kelly_diffuse_neutral_8k.png')
OUT_DIR = Path('demo_output/facial_expressions')
OUT_PATH = OUT_DIR / 'kelly_happy_8k.png'

mp_face_mesh = mp.solutions.face_mesh

# Known landmark indices for mouth corners in MediaPipe's 468-landmark topology
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

# We'll use the FACEMESH_LIPS set to capture mouth region points
try:
	from mediapipe.solutions.face_mesh_connections import FACEMESH_LIPS
	LIP_CONNECTIONS = FACEMESH_LIPS
except Exception:
	LIP_CONNECTIONS = []


def _unique_indices_from_connections(conns: List[Tuple[int, int]]) -> List[int]:
	idx = set()
	for a, b in conns:
		idx.add(a)
		idx.add(b)
	return sorted(idx)


def read_image_bgr(path: Path) -> np.ndarray:
	img = cv2.imread(str(path), cv2.IMREAD_COLOR)
	if img is None:
		raise FileNotFoundError(f"Could not read image: {path}")
	return img


def detect_landmarks(image_bgr: np.ndarray):
	# Use a downscaled copy for speed; keep scale to map back
	h, w = image_bgr.shape[:2]
	scale = 1280.0 / max(h, w)
	if scale < 1.0:
		ds = cv2.resize(image_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
	else:
		ds = image_bgr.copy()
	with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as face_mesh:
		res = face_mesh.process(cv2.cvtColor(ds, cv2.COLOR_BGR2RGB))
		if not res.multi_face_landmarks:
			raise RuntimeError('No face landmarks detected')
		landmarks = res.multi_face_landmarks[0].landmark
		# Map to original resolution
		inv = 1.0/scale if scale != 0 else 1.0
		points = np.array([[lm.x*ds.shape[1]*inv, lm.y*ds.shape[0]*inv] for lm in landmarks], dtype=np.float32)
		return points


def build_smile_targets(src_pts: np.ndarray, indices: List[int]) -> np.ndarray:
	"""
	Create destination points for a natural smile by raising mouth corners and slightly adjusting upper/lower lip.
	"""
	dst_pts = src_pts.copy()
	if len(indices) == 0:
		return dst_pts

	# Compute a reasonable offset relative to inter-corner distance
	lc = src_pts[LEFT_MOUTH_CORNER]
	rc = src_pts[RIGHT_MOUTH_CORNER]
	corner_dist = np.linalg.norm(rc - lc)
	off_up = max(4.0, corner_dist * 0.015)   # raise corners
	off_out = max(2.0, corner_dist * 0.01)   # pull outward a bit

	# Raise corners
	dst_pts[LEFT_MOUTH_CORNER] = [lc[0] - off_out, lc[1] - off_up]
	dst_pts[RIGHT_MOUTH_CORNER] = [rc[0] + off_out, rc[1] - off_up]

	# Nudge upper/lower lip rings using simple heuristic: move upper lip slightly up, lower lip slightly down
	for idx in indices:
		p = src_pts[idx]
		# Determine side: left/right helps give pleasant curl
		side = -1.0 if p[0] < (lc[0]+rc[0])/2 else 1.0
		# Vertical bias: pixels above mouth center treated as upper lip; below as lower lip
		center_y = (lc[1] + rc[1]) * 0.5
		if p[1] < center_y:
			vy = -off_up * 0.35
			vx = side * off_out * 0.15
		else:
			vy = off_up * 0.25
			vx = side * off_out * 0.1
		dst_pts[idx] = [p[0] + vx, p[1] + vy]

	return dst_pts


def piecewise_affine_warp_fullres(image_bgr: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, indices: List[int]) -> np.ndarray:
	"""
	Apply a localized piecewise-affine warp around the mouth region using skimage.
	We build a small mesh from the selected indices plus a padded ROI box to keep warp localized.
	"""
	h, w = image_bgr.shape[:2]
	# Determine ROI around all involved points
	sel = src_pts[indices]
	xmin, ymin = sel.min(axis=0)
	xmax, ymax = sel.max(axis=0)
	pad = max(20, int(0.06 * (xmax - xmin + ymax - ymin)))
	xmin = max(0, int(xmin - pad))
	ymin = max(0, int(ymin - pad))
	xmax = min(w - 1, int(xmax + pad))
	ymax = min(h - 1, int(ymax + pad))

	roi = image_bgr[ymin:ymax+1, xmin:xmax+1].copy()
	roi_h, roi_w = roi.shape[:2]
	if roi_h < 2 or roi_w < 2:
		return image_bgr

	# Translate points into ROI coords
	src_roi = src_pts[indices] - np.array([xmin, ymin], dtype=np.float32)
	dst_roi = dst_pts[indices] - np.array([xmin, ymin], dtype=np.float32)

	# Build a gentle grid over ROI and move only selected points; grid anchors keep rest stable
	# Create a coarse grid
	grid_rows, grid_cols = 12, 12
	grid_y = np.linspace(0, roi_h - 1, grid_rows)
	grid_x = np.linspace(0, roi_w - 1, grid_cols)
	gx, gy = np.meshgrid(grid_x, grid_y)
	grid_pts = np.column_stack([gx.flatten(), gy.flatten()]).astype(np.float32)

	# Concatenate with landmark points (landmarks influence warp; grid keeps stability)
	src_all = np.vstack([grid_pts, src_roi])
	dst_all = np.vstack([grid_pts, dst_roi])

	# Piecewise affine transform
	transform = PiecewiseAffineTransform()
	transform.estimate(dst_all, src_all)  # map dst->src for skimage.warp

	# skimage expects RGB float [0,1]
	roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) / 255.0
	warped_rgb = warp(roi_rgb, transform, output_shape=(roi_h, roi_w), mode='edge')
	warped_bgr = cv2.cvtColor((warped_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

	# Feathered composite back into original
	mask = np.zeros((roi_h, roi_w), dtype=np.float32)
	cv2.rectangle(mask, (0, 0), (roi_w - 1, roi_h - 1), 1.0, thickness=-1)
	feather = max(8, int(min(roi_h, roi_w) * 0.04))
	mask = cv2.GaussianBlur(mask, (feather | 1, feather | 1), 0)
	mask = mask[..., None]

	comp = (mask * warped_bgr + (1.0 - mask) * roi).astype(np.uint8)
	out = image_bgr.copy()
	out[ymin:ymax+1, xmin:xmax+1] = comp
	return out


def main():
	if not BASE_DIFFUSE.exists():
		print(f"Base texture not found: {BASE_DIFFUSE}", file=sys.stderr)
		sys.exit(1)
	OUT_DIR.mkdir(parents=True, exist_ok=True)

	img = read_image_bgr(BASE_DIFFUSE)
	src_pts = detect_landmarks(img)

	# Determine indices for lips region
	if LIP_CONNECTIONS:
		lip_indices = _unique_indices_from_connections(LIP_CONNECTIONS)
	else:
		# Fallback to a conservative set around mouth if connections missing
		lip_indices = sorted(set([LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER, 0, 13, 14, 17, 78, 308]))

	dst_pts = build_smile_targets(src_pts, lip_indices)
	out = piecewise_affine_warp_fullres(img, src_pts, dst_pts, lip_indices)

	cv2.imwrite(str(OUT_PATH), out)
	print(f"Generated: {OUT_PATH}")


if __name__ == '__main__':
	main()



























