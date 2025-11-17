#!/usr/bin/env python3
"""
Scan for all Kelly images and generate an HTML gallery
"""
import os
from pathlib import Path
from datetime import datetime

# Directories to scan (excluding Unity project files)
SCAN_DIRS = [
    'lessons',
    'iLearnStudio/projects/Kelly',
    'digital-kelly/assets/images',
    'lesson-player',
    'projects/Kelly',
    'synthetic_tts',
]

# Exclusions
EXCLUSIONS = [
    'digital-kelly/engines',
    'Library',
    'PackageCache',
    'node_modules',
    '.git',
    'backup',
    'test_comparison',
    'dist',
]

def is_kelly_image(path_str):
    """Check if image is related to Kelly"""
    path_lower = path_str.lower()
    return any(keyword in path_lower for keyword in ['kelly', 'curious'])

def should_exclude(path_str):
    """Check if path should be excluded"""
    return any(exclusion in path_str for exclusion in EXCLUSIONS)

def scan_images():
    """Scan for all Kelly images"""
    images = []
    root = Path('.')
    
    for scan_dir in SCAN_DIRS:
        scan_path = root / scan_dir
        if not scan_path.exists():
            continue
            
        for ext in ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']:
            for img_path in scan_path.rglob(f'*.{ext}'):
                path_str = str(img_path)
                if should_exclude(path_str):
                    continue
                if is_kelly_image(path_str) or scan_dir in ['lessons', 'lesson-player']:
                    try:
                        stat = img_path.stat()
                        images.append({
                            'path': path_str,
                            'size': stat.st_size,
                            'modified': datetime.fromtimestamp(stat.st_mtime),
                            'relative': str(img_path.relative_to(root))
                        })
                    except Exception as e:
                        print(f"Error reading {img_path}: {e}")
    
    return sorted(images, key=lambda x: x['relative'])

def format_size(size_bytes):
    """Format file size"""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"

def categorize_image(path):
    """Categorize image by location"""
    if 'lessons/images' in path:
        return 'Lesson Expressions'
    elif 'lessons' in path and path.endswith(('.png', '.PNG', '.jpg', '.jpeg')):
        return 'Lesson Assets'
    elif 'iLearnStudio' in path or 'projects/Kelly/Ref' in path:
        return 'Reference Images'
    elif 'lesson-player' in path:
        return 'Lesson Player'
    elif 'synthetic_tts' in path:
        return 'TTS Assets'
    elif 'projects/Kelly/assets' in path:
        return 'Production Assets'
    else:
        return 'Other'

def generate_html(images):
    """Generate HTML gallery"""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kelly Image Database - Complete Inventory</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .stats {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .stat {
            background: rgba(255,255,255,0.2);
            padding: 15px 25px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .content {
            padding: 40px;
        }
        .category {
            margin-bottom: 50px;
        }
        .category-header {
            background: #f8f9fa;
            padding: 20px;
            border-left: 5px solid #667eea;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        .category-title {
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 5px;
        }
        .category-count {
            color: #666;
            font-size: 0.9em;
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        .image-card {
            background: #f8f9fa;
            border-radius: 10px;
            overflow: hidden;
            transition: transform 0.3s, box-shadow 0.3s;
            border: 2px solid transparent;
        }
        .image-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            border-color: #667eea;
        }
        .image-wrapper {
            width: 100%;
            height: 200px;
            overflow: hidden;
            background: #e9ecef;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .image-wrapper img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            cursor: pointer;
            transition: transform 0.3s;
        }
        .image-wrapper img:hover {
            transform: scale(1.1);
        }
        .image-info {
            padding: 15px;
        }
        .image-name {
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
            word-break: break-word;
            font-size: 0.9em;
        }
        .image-details {
            font-size: 0.85em;
            color: #666;
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 5px;
        }
        .image-path {
            font-family: 'Courier New', monospace;
            font-size: 0.75em;
            color: #999;
            margin-top: 8px;
            word-break: break-all;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            overflow: auto;
        }
        .modal-content {
            margin: auto;
            display: block;
            max-width: 90%;
            max-height: 90vh;
            margin-top: 5vh;
        }
        .modal-close {
            position: absolute;
            top: 20px;
            right: 40px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }
        .modal-close:hover {
            color: #667eea;
        }
        .organization-section {
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 30px;
            margin-top: 40px;
            border-radius: 5px;
        }
        .organization-section h2 {
            color: #856404;
            margin-bottom: 20px;
        }
        .recommendation {
            background: white;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
            border-left: 3px solid #ffc107;
        }
        .recommendation h3 {
            color: #856404;
            margin-bottom: 10px;
        }
        .filter-bar {
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .filter-btn {
            padding: 8px 16px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .filter-btn:hover, .filter-btn.active {
            background: #667eea;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🖼️ Kelly Image Database</h1>
            <p style="font-size: 1.2em; margin-top: 10px;">Complete Inventory & Organization Analysis</p>
            <div class="stats">
                <div class="stat">
                    <div class="stat-number">{total_images}</div>
                    <div class="stat-label">Total Images</div>
                </div>
                <div class="stat">
                    <div class="stat-number">{total_size}</div>
                    <div class="stat-label">Total Size</div>
                </div>
                <div class="stat">
                    <div class="stat-number">{categories}</div>
                    <div class="stat-label">Categories</div>
                </div>
            </div>
        </header>
        <div class="content">
"""
    
    # Group images by category
    categorized = {}
    total_size = 0
    for img in images:
        category = categorize_image(img['path'])
        if category not in categorized:
            categorized[category] = []
        categorized[category].append(img)
        total_size += img['size']
    
    # Add gallery sections
    for category, imgs in sorted(categorized.items()):
        html += f"""
            <div class="category" data-category="{category.lower().replace(' ', '-')}">
                <div class="category-header">
                    <div class="category-title">{category}</div>
                    <div class="category-count">{len(imgs)} images</div>
                </div>
                <div class="gallery">
"""
        for img in imgs:
            html += f"""
                    <div class="image-card">
                        <div class="image-wrapper">
                            <img src="{img['relative']}" alt="{img['relative']}" onclick="openModal('{img['relative']}')" loading="lazy">
                        </div>
                        <div class="image-info">
                            <div class="image-name">{Path(img['relative']).name}</div>
                            <div class="image-details">
                                <span>{format_size(img['size'])}</span>
                                <span>{img['modified'].strftime('%Y-%m-%d')}</span>
                            </div>
                            <div class="image-path">{img['relative']}</div>
                        </div>
                    </div>
"""
        html += """
                </div>
            </div>
"""
    
    # Add organization recommendations
    html += """
            <div class="organization-section">
                <h2>📋 Organization Analysis & Recommendations</h2>
"""
    
    recommendations = []
    
    # Check for duplicates
    if 'lessons/Curious Kelly in final pose in Chair' in [img['relative'] for img in images]:
        recommendations.append({
            'title': 'Consolidate Duplicate Images',
            'text': 'Found duplicate images in different locations. Consider consolidating to a single source directory.'
        })
    
    # Check reference images location
    ref_images = [img for img in images if 'Ref' in img['path']]
    if len(ref_images) > 3:
        recommendations.append({
            'title': 'Reference Images Organization',
            'text': f'Found {len(ref_images)} reference images. Consider keeping all reference images in one canonical location: <code>iLearnStudio/projects/Kelly/Ref/</code>'
        })
    
    # Check lesson images
    lesson_images = [img for img in images if 'lessons' in img['path']]
    if len(lesson_images) > 10:
        recommendations.append({
            'title': 'Lesson Images Structure',
            'text': f'Found {len(lesson_images)} lesson-related images. Current structure is good, but consider: <ul><li>Keep all expression images in <code>lessons/images/</code></li><li>Move main lesson assets to <code>lessons/assets/</code></li><li>Create subdirectories by purpose (expressions, backgrounds, etc.)</li></ul>'
        })
    
    recommendations.append({
        'title': 'Proposed Directory Structure',
        'text': '''
        <strong>Recommended organization:</strong>
        <ul>
            <li><code>lessons/images/expressions/</code> - All Kelly expression images (director's chair poses)</li>
            <li><code>lessons/images/zoom-levels/</code> - Zoom level images (close-up, head-shoulders, etc.)</li>
            <li><code>iLearnStudio/projects/Kelly/Ref/</code> - All reference images (front, profile, three-quarter)</li>
            <li><code>projects/Kelly/assets/renders/</code> - Production renders and identity sheets</li>
            <li><code>lesson-player/assets/</code> - Player-specific assets</li>
        </ul>
        '''
    })
    
    recommendations.append({
        'title': 'Image Naming Convention',
        'text': '''
        <strong>Current issues:</strong>
        <ul>
            <li>Mixed case (PNG vs png)</li>
            <li>Spaces in filenames</li>
            <li>Inconsistent naming patterns</li>
        </ul>
        <strong>Recommended:</strong>
        <ul>
            <li>Use lowercase: <code>kelly-directors-chair-curious.png</code></li>
            <li>Use hyphens instead of spaces</li>
            <li>Include purpose/context: <code>kelly-zoom-level-1-head-shoulders.png</code></li>
        </ul>
        '''
    })
    
    for rec in recommendations:
        html += f"""
                <div class="recommendation">
                    <h3>{rec['title']}</h3>
                    <div>{rec['text']}</div>
                </div>
"""
    
    html += """
            </div>
        </div>
    </div>
    
    <div id="imageModal" class="modal">
        <span class="modal-close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>
    
    <script>
        function openModal(imagePath) {
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            modal.style.display = 'block';
            modalImg.src = imagePath;
        }
        
        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }
        
        window.onclick = function(event) {
            const modal = document.getElementById('imageModal');
            if (event.target == modal) {
                closeModal();
            }
        }
    </script>
</body>
</html>
"""
    
    # Format stats - replace placeholders manually to avoid CSS brace conflicts
    html = html.replace('{total_images}', str(len(images)))
    html = html.replace('{total_size}', format_size(total_size))
    html = html.replace('{categories}', str(len(categorized)))
    
    return html

if __name__ == '__main__':
    print("Scanning for Kelly images...")
    images = scan_images()
    print(f"Found {len(images)} images")
    
    print("Generating HTML gallery...")
    html = generate_html(images)
    
    output_file = 'kelly_image_database.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Generated {output_file}")
    print(f"   Total images: {len(images)}")
    print(f"   Categories: {len(set(categorize_image(img['path']) for img in images))}")

