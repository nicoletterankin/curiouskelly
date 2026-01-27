"""
Automatic LED path detection for the Ziggurat building.
Detects horizontal terrace edges where LED strips would be installed.
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path

# Paths
SOURCE_DIR = Path("public/ziggurat-transforms")
OUTPUT_FILE = Path("public/ziggurat-led-paths.json")

# Source images to process
SOURCE_IMAGES = [
    {'id': 'loc-01700', 'file': 'source-loc-01700.jpg'},
    {'id': 'loc-01701', 'file': 'source-loc-01701.jpg'},
    {'id': 'loc-01702', 'file': 'source-loc-01702.jpg'},
    {'id': 'loc-01703', 'file': 'source-loc-01703.jpg'},
    {'id': 'loc-01704', 'file': 'source-loc-01704.jpg'},
    {'id': 'loc-01708', 'file': 'source-loc-01708.jpg'},
    {'id': 'oc-1975-aerial', 'file': 'source-oc-1975-aerial.jpg'},
    {'id': 'oc-1980s-reservoir', 'file': 'source-oc-1980s-reservoir.jpg'},
    {'id': 'oc-1985-exterior', 'file': 'source-oc-1985-exterior.jpg'},
    {'id': 'oc-1985-facade', 'file': 'source-oc-1985-facade.jpg'},
    {'id': 'oc-1985-wide', 'file': 'source-oc-1985-wide.jpg'},
    {'id': 'oc-1985-approach', 'file': 'source-oc-1985-approach.jpg'},
    {'id': 'oc-1985-side', 'file': 'source-oc-1985-side.jpg'},
    {'id': 'oc-1985-panorama', 'file': 'source-oc-1985-panorama.jpg'},
    {'id': 'oc-1989-view1', 'file': 'source-oc-1989-view1.jpg'},
    {'id': 'oc-1989-view2', 'file': 'source-oc-1989-view2.jpg'},
    {'id': 'modern-wikimedia', 'file': 'source-modern-wikimedia-2020.jpg'},
    {'id': 'modern-gsa', 'file': 'source-modern-gsa-official.jpg'},
]

def detect_terrace_edges(image_path, image_id):
    """
    Detect horizontal terrace edges in a building image.
    Returns list of paths (each path is list of {x, y} points).
    """
    print(f"Processing: {image_id}")
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  Could not load: {image_path}")
        return []
    
    height, width = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection with Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Detect lines using Hough Transform
    # Focus on near-horizontal lines (terrace edges)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=100,
        minLineLength=width * 0.15,  # At least 15% of image width
        maxLineGap=20
    )
    
    if lines is None:
        print(f"  No lines detected")
        return []
    
    # Filter for horizontal lines (angle < 15 degrees from horizontal)
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate angle
        if x2 - x1 == 0:
            continue
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        # Keep lines that are nearly horizontal (within 15 degrees)
        if angle < 15 or angle > 165:
            # Ensure x1 < x2 for consistency
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            horizontal_lines.append((x1, y1, x2, y2))
    
    print(f"  Found {len(horizontal_lines)} horizontal lines")
    
    # Merge nearby lines (within 20px vertically)
    merged_lines = merge_nearby_lines(horizontal_lines, y_threshold=20)
    print(f"  After merging: {len(merged_lines)} lines")
    
    # Convert to path format for compositor
    paths = []
    for x1, y1, x2, y2 in merged_lines:
        # Each line becomes a 2-point path
        path = [
            {'x': float(x1), 'y': float(y1)},
            {'x': float(x2), 'y': float(y2)}
        ]
        paths.append(path)
    
    return paths


def merge_nearby_lines(lines, y_threshold=20):
    """
    Merge horizontal lines that are close together vertically.
    """
    if not lines:
        return []
    
    # Sort by y-coordinate (top to bottom)
    sorted_lines = sorted(lines, key=lambda l: (l[1] + l[3]) / 2)
    
    merged = []
    current_group = [sorted_lines[0]]
    
    for line in sorted_lines[1:]:
        # Average y of current group
        group_y = np.mean([(l[1] + l[3]) / 2 for l in current_group])
        line_y = (line[1] + line[3]) / 2
        
        if abs(line_y - group_y) < y_threshold:
            # Add to current group
            current_group.append(line)
        else:
            # Merge current group and start new one
            merged.append(merge_line_group(current_group))
            current_group = [line]
    
    # Don't forget last group
    if current_group:
        merged.append(merge_line_group(current_group))
    
    return merged


def merge_line_group(lines):
    """
    Merge a group of nearby lines into one line spanning the full extent.
    """
    min_x = min(min(l[0], l[2]) for l in lines)
    max_x = max(max(l[0], l[2]) for l in lines)
    avg_y = np.mean([(l[1] + l[3]) / 2 for l in lines])
    
    return (min_x, avg_y, max_x, avg_y)


def main():
    all_paths = {}
    
    for img_info in SOURCE_IMAGES:
        image_path = SOURCE_DIR / img_info['file']
        
        if not image_path.exists():
            print(f"Skipping (not found): {image_path}")
            continue
        
        paths = detect_terrace_edges(image_path, img_info['id'])
        
        if paths:
            all_paths[img_info['id']] = paths
    
    # Save to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_paths, f, indent=2)
    
    print(f"\nSaved {len(all_paths)} image paths to {OUTPUT_FILE}")
    
    # Summary
    total_paths = sum(len(p) for p in all_paths.values())
    print(f"Total LED paths detected: {total_paths}")


if __name__ == "__main__":
    main()
