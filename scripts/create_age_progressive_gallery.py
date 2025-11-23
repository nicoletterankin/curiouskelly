#!/usr/bin/env python3
"""
Create HTML Review Gallery for Age-Progressive Kelly Images
Generates interactive gallery for reviewing generated images
"""

from pathlib import Path
from typing import List, Dict
import json


# Age and pose definitions
AGE_GROUPS = {
    "3": {"range": "2-5", "persona": "Playful Toddler"},
    "9": {"range": "6-12", "persona": "Curious Kid"},
    "15": {"range": "13-17", "persona": "Enthusiastic Teen"},
    "27": {"range": "18-35", "persona": "Knowledgeable Adult"},
    "48": {"range": "36-60", "persona": "Wise Mentor"},
    "82": {"range": "61-102", "persona": "Reflective Elder"}
}

POSES = {
    "pose1": "Full Body Seated",
    "pose2": "Upper Body Seated",
    "pose3": "Close-up Portrait",
    "pose4": "Front-Facing Lean"
}

FORMATS = {
    "16x9": "16:9 (Lesson Player)",
    "1x1": "1:1 (3D Reference)",
    "3x4": "3:4 (Portrait)"
}


def scan_generated_images(renders_dir: Path) -> Dict:
    """Scan renders directory and catalog all generated images"""
    catalog = {}
    
    for age in AGE_GROUPS.keys():
        catalog[age] = {}
        for pose in POSES.keys():
            catalog[age][pose] = {}
            for format_key in FORMATS.keys():
                filename = f"kelly_age{age}_{pose}_{format_key}.png"
                filepath = renders_dir / filename
                catalog[age][pose][format_key] = {
                    "exists": filepath.exists(),
                    "path": f"renders/{filename}" if filepath.exists() else None,
                    "filename": filename
                }
    
    return catalog


def generate_html_gallery(output_path: Path, renders_dir: Path):
    """Generate interactive HTML review gallery"""
    
    catalog = scan_generated_images(renders_dir)
    
    # Count statistics
    total_expected = len(AGE_GROUPS) * len(POSES) * len(FORMATS)
    total_generated = sum(
        1 for age in catalog.values()
        for pose in age.values()
        for format_data in pose.values()
        if format_data["exists"]
    )
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kelly Age-Progressive Image Gallery</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        
        header {{
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 20px;
        }}
        
        h1 {{
            font-size: 2.5em;
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 20px 0;
            font-size: 1.1em;
        }}
        
        .stat {{
            padding: 10px 20px;
            background: #f0f4ff;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }}
        
        .stat strong {{
            color: #667eea;
        }}
        
        .controls {{
            display: flex;
            gap: 15px;
            margin-bottom: 30px;
            flex-wrap: wrap;
            justify-content: center;
        }}
        
        .control-group {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
        }}
        
        .control-group label {{
            font-weight: 600;
            margin-right: 10px;
            color: #555;
        }}
        
        select, button {{
            padding: 8px 15px;
            border-radius: 5px;
            border: 2px solid #667eea;
            font-size: 14px;
            cursor: pointer;
        }}
        
        button {{
            background: #667eea;
            color: white;
            font-weight: 600;
            transition: all 0.3s;
        }}
        
        button:hover {{
            background: #764ba2;
            transform: translateY(-2px);
        }}
        
        .age-section {{
            margin-bottom: 50px;
        }}
        
        .age-header {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        
        .age-header h2 {{
            font-size: 1.8em;
        }}
        
        .age-header p {{
            opacity: 0.9;
            margin-top: 5px;
        }}
        
        .pose-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .pose-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border: 2px solid #e0e0e0;
            transition: all 0.3s;
        }}
        
        .pose-card:hover {{
            border-color: #667eea;
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.2);
            transform: translateY(-5px);
        }}
        
        .pose-card h3 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.2em;
        }}
        
        .format-tabs {{
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
            border-bottom: 2px solid #e0e0e0;
        }}
        
        .format-tab {{
            padding: 8px 15px;
            cursor: pointer;
            border-radius: 5px 5px 0 0;
            transition: all 0.2s;
            font-size: 0.9em;
        }}
        
        .format-tab:hover {{
            background: #e8ecff;
        }}
        
        .format-tab.active {{
            background: #667eea;
            color: white;
            font-weight: 600;
        }}
        
        .image-container {{
            position: relative;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            min-height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .image-container img {{
            max-width: 100%;
            height: auto;
            display: block;
        }}
        
        .missing {{
            color: #999;
            text-align: center;
            padding: 40px;
            font-style: italic;
        }}
        
        .missing::before {{
            content: "⚠️";
            display: block;
            font-size: 3em;
            margin-bottom: 10px;
        }}
        
        .checklist {{
            background: #fffbf0;
            border: 2px solid #ffc107;
            border-radius: 10px;
            padding: 20px;
            margin-top: 40px;
        }}
        
        .checklist h2 {{
            color: #f57c00;
            margin-bottom: 15px;
        }}
        
        .checklist-item {{
            padding: 10px;
            margin: 5px 0;
            background: white;
            border-radius: 5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .checklist-item input[type="checkbox"] {{
            width: 20px;
            height: 20px;
            cursor: pointer;
        }}
        
        .hidden {{
            display: none !important;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎨 Kelly Age-Progressive Image Gallery</h1>
            <p>Review and validate generated images across 6 ages, 4 poses, and 3 formats</p>
            <div class="stats">
                <div class="stat">
                    <strong>Generated:</strong> {total_generated}/{total_expected}
                </div>
                <div class="stat">
                    <strong>Success Rate:</strong> {(total_generated/total_expected*100):.1f}%
                </div>
            </div>
        </header>
        
        <div class="controls">
            <div class="control-group">
                <label for="ageFilter">Filter by Age:</label>
                <select id="ageFilter">
                    <option value="all">All Ages</option>
                    {''.join(f'<option value="{age}">Age {age} ({AGE_GROUPS[age]["persona"]})</option>' for age in AGE_GROUPS.keys())}
                </select>
            </div>
            <div class="control-group">
                <label for="poseFilter">Filter by Pose:</label>
                <select id="poseFilter">
                    <option value="all">All Poses</option>
                    {''.join(f'<option value="{pose}">{POSES[pose]}</option>' for pose in POSES.keys())}
                </select>
            </div>
            <div class="control-group">
                <button onclick="resetFilters()">Reset Filters</button>
            </div>
        </div>
        
        <div id="gallery">
"""
    
    # Generate age sections
    for age, age_info in AGE_GROUPS.items():
        html += f"""
            <div class="age-section" data-age="{age}">
                <div class="age-header">
                    <h2>Kelly at Age {age}</h2>
                    <p>Age Range: {age_info['range']} | Persona: {age_info['persona']}</p>
                </div>
                <div class="pose-grid">
"""
        
        # Generate pose cards
        for pose, pose_name in POSES.items():
            pose_data = catalog[age][pose]
            
            html += f"""
                    <div class="pose-card" data-pose="{pose}">
                        <h3>{pose_name}</h3>
                        <div class="format-tabs">
"""
            
            # Format tabs
            for idx, (format_key, format_name) in enumerate(FORMATS.items()):
                active_class = "active" if idx == 0 else ""
                html += f'                            <div class="format-tab {active_class}" onclick="switchFormat(this, \'{age}_{pose}_{format_key}\')">{format_name}</div>\n'
            
            html += """                        </div>
                        <div class="image-container">
"""
            
            # Format images
            for idx, (format_key, format_name) in enumerate(FORMATS.items()):
                display = "block" if idx == 0 else "none"
                img_data = pose_data[format_key]
                
                if img_data["exists"]:
                    html += f'                            <img id="{age}_{pose}_{format_key}" src="{img_data["path"]}" alt="{format_name}" style="display: {display};">\n'
                else:
                    html += f'                            <div id="{age}_{pose}_{format_key}" class="missing" style="display: {display};">Image not generated<br><small>{img_data["filename"]}</small></div>\n'
            
            html += """                        </div>
                    </div>
"""
        
        html += """                </div>
            </div>
"""
    
    # Validation checklist
    html += """
        </div>
        
        <div class="checklist">
            <h2>✅ Quality Validation Checklist</h2>
            <p>Review each image set and check off items as you validate:</p>
            <div class="checklist-item">
                <input type="checkbox" id="check1">
                <label for="check1">All 72 images generated successfully</label>
            </div>
            <div class="checklist-item">
                <input type="checkbox" id="check2">
                <label for="check2">Kelly's identity recognizable across all ages</label>
            </div>
            <div class="checklist-item">
                <input type="checkbox" id="check3">
                <label for="check3">Aging progression looks natural and believable</label>
            </div>
            <div class="checklist-item">
                <input type="checkbox" id="check4">
                <label for="check4">Poses are consistent across age groups</label>
            </div>
            <div class="checklist-item">
                <input type="checkbox" id="check5">
                <label for="check5">Blue sweater visible and consistent in all images</label>
            </div>
            <div class="checklist-item">
                <input type="checkbox" id="check6">
                <label for="check6">Director's chair visible in poses 1, 2, and 4</label>
            </div>
            <div class="checklist-item">
                <input type="checkbox" id="check7">
                <label for="check7">Warm engaging smile consistent across all images</label>
            </div>
            <div class="checklist-item">
                <input type="checkbox" id="check8">
                <label for="check8">Background and lighting consistent (white studio with geometric shadows)</label>
            </div>
            <div class="checklist-item">
                <input type="checkbox" id="check9">
                <label for="check9">All three aspect ratios properly formatted (16:9, 1:1, 3:4)</label>
            </div>
            <div class="checklist-item">
                <input type="checkbox" id="check10">
                <label for="check10">No obvious artifacts, distortions, or anatomical issues</label>
            </div>
        </div>
    </div>
    
    <script>
        function switchFormat(tab, imageId) {
            // Get parent pose card
            const poseCard = tab.closest('.pose-card');
            
            // Hide all images in this card
            const images = poseCard.querySelectorAll('.image-container > *');
            images.forEach(img => img.style.display = 'none');
            
            // Show selected image
            const targetImage = document.getElementById(imageId);
            if (targetImage) {
                targetImage.style.display = 'block';
            }
            
            // Update active tab
            const tabs = poseCard.querySelectorAll('.format-tab');
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
        }
        
        function resetFilters() {
            document.getElementById('ageFilter').value = 'all';
            document.getElementById('poseFilter').value = 'all';
            filterGallery();
        }
        
        function filterGallery() {
            const ageFilter = document.getElementById('ageFilter').value;
            const poseFilter = document.getElementById('poseFilter').value;
            
            // Filter age sections
            document.querySelectorAll('.age-section').forEach(section => {
                const age = section.dataset.age;
                const ageMatch = ageFilter === 'all' || ageFilter === age;
                section.classList.toggle('hidden', !ageMatch);
            });
            
            // Filter pose cards
            document.querySelectorAll('.pose-card').forEach(card => {
                const pose = card.dataset.pose;
                const poseMatch = poseFilter === 'all' || poseFilter === pose;
                card.classList.toggle('hidden', !poseMatch);
            });
        }
        
        // Add event listeners
        document.getElementById('ageFilter').addEventListener('change', filterGallery);
        document.getElementById('poseFilter').addEventListener('change', filterGallery);
        
        // Save checklist state to localStorage
        document.querySelectorAll('.checklist-item input[type="checkbox"]').forEach(checkbox => {
            const id = checkbox.id;
            
            // Restore saved state
            const saved = localStorage.getItem(id);
            if (saved === 'true') {
                checkbox.checked = true;
            }
            
            // Save on change
            checkbox.addEventListener('change', () => {
                localStorage.setItem(id, checkbox.checked);
            });
        });
    </script>
</body>
</html>
"""
    
    # Write HTML file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')
    print(f"✅ Generated gallery: {output_path}")
    print(f"   Total images: {total_generated}/{total_expected}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create HTML review gallery for age-progressive images")
    parser.add_argument("--output", type=Path, default=Path("projects/Kelly/assets/age_progressive/review.html"))
    parser.add_argument("--renders-dir", type=Path, default=Path("projects/Kelly/assets/age_progressive/renders"))
    
    args = parser.parse_args()
    
    generate_html_gallery(args.output, args.renders_dir)


if __name__ == "__main__":
    main()



