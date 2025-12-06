#!/usr/bin/env python3
"""
KELLY COP - Face Recognition Audit Tool
========================================
Scans all Kelly images and verifies faces match "Our Kelly" using DeepFace.

This is a more accurate version that compares FACES, not entire images.

Usage:
    python kelly_face_audit.py [--threshold 0.6] [--html]

Dependencies:
    pip install deepface opencv-python tf-keras Pillow rich
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any

# Force UTF-8 for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

try:
    from PIL import Image
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.panel import Panel
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install Pillow rich")
    sys.exit(1)

console = Console(force_terminal=True, legacy_windows=True)

# Lazy load deepface to avoid startup warnings
DeepFace = None

def load_deepface():
    global DeepFace
    if DeepFace is None:
        console.print("[dim]Loading face recognition models...[/dim]")
        from deepface import DeepFace as DF
        DeepFace = DF
    return DeepFace

# =============================================================================
# CONFIGURATION
# =============================================================================

# Canonical reference folder
CANONICAL_REF_PATH = Path(r"C:\iLearnStudio\projects\Kelly\Ref\Best Character Reference")

# Best reference images for face comparison (close-ups work best)
BEST_FACE_REFS = [
    "close up of face.jpeg",
    "head and shoulders without chair.png",
    "neutral face with hair.png",
]

# Folders to scan for Kelly images
SCAN_FOLDERS = [
    Path(r"C:\Users\user\UI-TARS-desktop\public\kelly"),
    Path(r"C:\Users\user\UI-TARS-desktop\generated-poses-production"),
    Path(r"C:\Users\user\UI-TARS-desktop\generated-poses-presenter"),
]

# Root-level test files
ROOT_KELLY_FILES = [
    Path(r"C:\Users\user\UI-TARS-desktop\kelly-lora-test.png"),
    Path(r"C:\Users\user\UI-TARS-desktop\test_kelly_character_consistent.png"),
]

# Output paths
REPORT_PATH = Path(r"C:\Users\user\UI-TARS-desktop\tools\kelly-cop\face_audit_report")

# Image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}

# Face verification threshold (0-1, lower = stricter)
DEFAULT_THRESHOLD = 0.55


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FaceAuditResult:
    """Result of face-based audit for a single image"""
    file_path: str
    file_name: str
    folder: str
    face_detected: bool
    min_distance: float
    closest_reference: str
    status: str  # MATCH, SUSPICIOUS, NO_MATCH, NO_FACE, ERROR
    confidence: float
    all_distances: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class FaceAuditReport:
    """Complete face audit report"""
    timestamp: str
    threshold_used: float
    total_images: int = 0
    faces_detected: int = 0
    matches: int = 0
    suspicious: int = 0
    no_matches: int = 0
    no_faces: int = 0
    errors: int = 0
    results: List[FaceAuditResult] = field(default_factory=list)


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def verify_face(img_path: str, ref_path: str, model_name: str = "VGG-Face") -> Tuple[bool, float]:
    """Verify if two images contain the same face."""
    try:
        df = load_deepface()
        result = df.verify(
            img1_path=img_path,
            img2_path=ref_path,
            model_name=model_name,
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
        )
        return True, result['distance']
    except Exception as e:
        return False, 1.0


def load_reference_images() -> Dict[str, str]:
    """Load paths to reference images"""
    refs = {}
    
    console.print(f"\n[cyan]Loading face references from:[/cyan] {CANONICAL_REF_PATH}")
    
    for ref_name in BEST_FACE_REFS:
        ref_path = CANONICAL_REF_PATH / ref_name
        if ref_path.exists():
            refs[ref_name] = str(ref_path)
            console.print(f"  [green]+[/green] {ref_name} [dim](primary)[/dim]")
    
    for file in CANONICAL_REF_PATH.iterdir():
        if file.suffix.lower() in IMAGE_EXTENSIONS and file.name not in refs:
            refs[file.name] = str(file)
            console.print(f"  [dim]+ {file.name} (backup)[/dim]")
    
    console.print(f"[green]Loaded {len(refs)} face references[/green]\n")
    return refs


def collect_images(folders: List[Path], root_files: List[Path], limit: Optional[int] = None) -> List[Path]:
    """Collect image files to audit"""
    files = []
    
    for f in root_files:
        if f.exists() and f.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(f)
    
    for folder in folders:
        if folder.exists():
            for ext in IMAGE_EXTENSIONS:
                files.extend(folder.rglob(f"*{ext}"))
    
    if limit:
        files = files[:limit]
    
    return files


def audit_image_face(
    file_path: Path,
    references: Dict[str, str],
    threshold: float
) -> FaceAuditResult:
    """Audit a single image using face recognition"""
    
    result = FaceAuditResult(
        file_path=str(file_path),
        file_name=file_path.name,
        folder=str(file_path.parent),
        face_detected=False,
        min_distance=1.0,
        closest_reference="none",
        status="ERROR",
        confidence=0,
    )
    
    try:
        distances = {}
        for ref_name, ref_path in references.items():
            success, distance = verify_face(str(file_path), ref_path)
            if success:
                distances[ref_name] = distance
        
        if not distances:
            result.status = "NO_FACE"
            result.error = "No face detected in image"
            return result
        
        result.face_detected = True
        result.all_distances = distances
        
        min_ref = min(distances.keys(), key=lambda k: distances[k])
        min_dist = distances[min_ref]
        
        result.min_distance = min_dist
        result.closest_reference = min_ref
        result.confidence = max(0, min(100, (1 - min_dist) * 100))
        
        if min_dist <= threshold * 0.7:
            result.status = "MATCH"
        elif min_dist <= threshold:
            result.status = "SUSPICIOUS"
        else:
            result.status = "NO_MATCH"
        
    except Exception as e:
        result.error = str(e)
        result.status = "ERROR"
    
    return result


def run_face_audit(
    references: Dict[str, str],
    threshold: float,
    limit: Optional[int] = None
) -> List[FaceAuditResult]:
    """Run face audit on all images"""
    
    console.print("[cyan]Collecting images to audit...[/cyan]")
    files = collect_images(SCAN_FOLDERS, ROOT_KELLY_FILES, limit)
    console.print(f"[green]Found {len(files)} images[/green]\n")
    
    if not files:
        return []
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing faces...", total=len(files))
        
        for file in files:
            result = audit_image_face(file, references, threshold)
            results.append(result)
            progress.update(task, advance=1)
    
    return results


def generate_report(results: List[FaceAuditResult], threshold: float) -> FaceAuditReport:
    """Generate audit report"""
    report = FaceAuditReport(
        timestamp=datetime.now().isoformat(),
        threshold_used=threshold,
        total_images=len(results),
    )
    
    for r in results:
        report.results.append(r)
        
        if r.face_detected:
            report.faces_detected += 1
        
        if r.status == "MATCH":
            report.matches += 1
        elif r.status == "SUSPICIOUS":
            report.suspicious += 1
        elif r.status == "NO_MATCH":
            report.no_matches += 1
        elif r.status == "NO_FACE":
            report.no_faces += 1
        else:
            report.errors += 1
    
    return report


def print_summary(report: FaceAuditReport):
    """Print summary to console"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]KELLY COP FACE AUDIT COMPLETE[/bold cyan]",
        border_style="cyan"
    ))
    
    console.print(f"\n[bold]SUMMARY[/bold]")
    console.print(f"  Total images:    [cyan]{report.total_images}[/cyan]")
    console.print(f"  Faces detected:  [cyan]{report.faces_detected}[/cyan]")
    console.print(f"  Threshold used:  [cyan]{report.threshold_used}[/cyan]")
    console.print(f"  [OK] Matches:       [green]{report.matches}[/green]")
    console.print(f"  [!!] Suspicious:    [yellow]{report.suspicious}[/yellow]")
    console.print(f"  [XX] No Match:      [red]{report.no_matches}[/red]")
    console.print(f"  [??] No Face:       [dim]{report.no_faces}[/dim]")
    console.print(f"  [ER] Errors:        [dim]{report.errors}[/dim]")
    
    no_matches = [r for r in report.results if r.status == "NO_MATCH"]
    if no_matches:
        console.print(f"\n[bold red]NO MATCH FILES ({len(no_matches)}):[/bold red]")
        for r in no_matches[:20]:
            console.print(f"  [red]* {r.file_name}[/red] (dist: {r.min_distance:.3f})")
        if len(no_matches) > 20:
            console.print(f"  [dim]... and {len(no_matches) - 20} more[/dim]")


def save_report(report: FaceAuditReport):
    """Save report to files"""
    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_file = REPORT_PATH / f"face_audit_{timestamp}.json"
    with open(summary_file, 'w') as f:
        summary = {
            'timestamp': report.timestamp,
            'threshold_used': report.threshold_used,
            'total_images': report.total_images,
            'faces_detected': report.faces_detected,
            'matches': report.matches,
            'suspicious': report.suspicious,
            'no_matches': report.no_matches,
            'no_faces': report.no_faces,
            'errors': report.errors,
        }
        json.dump(summary, f, indent=2)
    console.print(f"\n[green]Summary saved to:[/green] {summary_file}")
    
    csv_file = REPORT_PATH / f"face_audit_details_{timestamp}.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("file_path,file_name,folder,status,confidence,min_distance,closest_reference,face_detected\n")
        for r in report.results:
            f.write(f'"{r.file_path}","{r.file_name}","{r.folder}",{r.status},{r.confidence:.1f},{r.min_distance:.4f},"{r.closest_reference}",{r.face_detected}\n')
    console.print(f"[green]Details saved to:[/green] {csv_file}")
    
    no_matches = [r for r in report.results if r.status == "NO_MATCH"]
    if no_matches:
        no_match_file = REPORT_PATH / f"face_no_matches_{timestamp}.txt"
        with open(no_match_file, 'w') as f:
            for r in no_matches:
                f.write(f"{r.file_path}\n")
        console.print(f"[red]No-match list saved to:[/red] {no_match_file}")
    
    return csv_file


def generate_html_report(report: FaceAuditReport):
    """Generate visual HTML report"""
    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_file = REPORT_PATH / f"face_visual_report_{timestamp}.html"
    
    matches = [r for r in report.results if r.status == "MATCH"][:30]
    suspicious = [r for r in report.results if r.status == "SUSPICIOUS"][:30]
    no_matches = [r for r in report.results if r.status == "NO_MATCH"]
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Kelly Cop Face Audit - {timestamp}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }}
        h1 {{ color: #00d9ff; text-align: center; font-size: 2.5em; margin-bottom: 10px; }}
        h2 {{ color: #ffd700; border-bottom: 2px solid #ffd700; padding-bottom: 10px; margin-top: 40px; }}
        .subtitle {{ text-align: center; opacity: 0.7; margin-bottom: 30px; }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
            margin: 30px 0;
        }}
        .stat {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px 30px;
            text-align: center;
            min-width: 120px;
        }}
        .stat.match {{ border: 2px solid #00ff88; }}
        .stat.suspicious {{ border: 2px solid #ffaa00; }}
        .stat.nomatch {{ border: 2px solid #ff4444; }}
        .stat-num {{ font-size: 36px; font-weight: bold; }}
        .stat-label {{ font-size: 12px; opacity: 0.8; margin-top: 5px; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            overflow: hidden;
            transition: transform 0.2s;
        }}
        .card:hover {{ transform: translateY(-5px); }}
        .card.match {{ border: 3px solid #00ff88; }}
        .card.suspicious {{ border: 3px solid #ffaa00; }}
        .card.nomatch {{ border: 3px solid #ff4444; }}
        .card img {{
            width: 100%;
            height: 200px;
            object-fit: cover;
            background: #0f0f23;
        }}
        .card-body {{ padding: 15px; }}
        .card-title {{
            font-size: 12px;
            word-break: break-all;
            opacity: 0.8;
            margin-bottom: 10px;
        }}
        .card-stats {{ display: flex; justify-content: space-between; align-items: center; }}
        .confidence {{ font-size: 24px; font-weight: bold; }}
        .confidence.high {{ color: #00ff88; }}
        .confidence.medium {{ color: #ffaa00; }}
        .confidence.low {{ color: #ff4444; }}
    </style>
</head>
<body>
    <h1>KELLY COP FACE AUDIT</h1>
    <p class="subtitle">Generated: {timestamp} | Threshold: {report.threshold_used}</p>
    
    <div class="stats">
        <div class="stat">
            <div class="stat-num" style="color: #00d9ff;">{report.total_images}</div>
            <div class="stat-label">Total Images</div>
        </div>
        <div class="stat match">
            <div class="stat-num" style="color: #00ff88;">{report.matches}</div>
            <div class="stat-label">Matches</div>
        </div>
        <div class="stat suspicious">
            <div class="stat-num" style="color: #ffaa00;">{report.suspicious}</div>
            <div class="stat-label">Suspicious</div>
        </div>
        <div class="stat nomatch">
            <div class="stat-num" style="color: #ff4444;">{report.no_matches}</div>
            <div class="stat-label">No Match</div>
        </div>
    </div>
"""
    
    if no_matches:
        html += f"""
    <h2>NO MATCH - Different Person ({len(no_matches)} files)</h2>
    <div class="grid">
"""
        for r in no_matches:
            file_path = r.file_path.replace('\\', '/')
            html += f"""
        <div class="card nomatch">
            <img src="file:///{file_path}" onerror="this.style.background='#333'">
            <div class="card-body">
                <div class="card-title">{r.file_name}</div>
                <div class="card-stats">
                    <span class="confidence low">{r.confidence:.0f}%</span>
                    <span style="font-size:12px;opacity:0.7">Dist: {r.min_distance:.3f}</span>
                </div>
            </div>
        </div>
"""
        html += "    </div>\n"
    
    if suspicious:
        html += f"""
    <h2>SUSPICIOUS - Needs Review ({len(suspicious)} shown)</h2>
    <div class="grid">
"""
        for r in suspicious:
            file_path = r.file_path.replace('\\', '/')
            html += f"""
        <div class="card suspicious">
            <img src="file:///{file_path}" onerror="this.style.background='#333'">
            <div class="card-body">
                <div class="card-title">{r.file_name}</div>
                <div class="card-stats">
                    <span class="confidence medium">{r.confidence:.0f}%</span>
                    <span style="font-size:12px;opacity:0.7">Dist: {r.min_distance:.3f}</span>
                </div>
            </div>
        </div>
"""
        html += "    </div>\n"
    
    if matches:
        html += f"""
    <h2>MATCHES - Our Kelly ({len(matches)} shown of {report.matches})</h2>
    <div class="grid">
"""
        for r in matches:
            file_path = r.file_path.replace('\\', '/')
            html += f"""
        <div class="card match">
            <img src="file:///{file_path}" onerror="this.style.background='#333'">
            <div class="card-body">
                <div class="card-title">{r.file_name}</div>
                <div class="card-stats">
                    <span class="confidence high">{r.confidence:.0f}%</span>
                    <span style="font-size:12px;opacity:0.7">Dist: {r.min_distance:.3f}</span>
                </div>
            </div>
        </div>
"""
        html += "    </div>\n"
    
    html += """
</body>
</html>
"""
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    console.print(f"[green]Visual report saved to:[/green] {html_file}")
    return html_file


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Kelly Cop Face Recognition Audit")
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                       help=f"Face match threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument('--limit', type=int, default=None,
                       help="Limit number of images to scan (for testing)")
    parser.add_argument('--html', action='store_true',
                       help="Generate HTML visual report")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]KELLY COP[/bold cyan]\n"
        "[dim]Face Recognition Audit[/dim]",
        border_style="cyan"
    ))
    
    references = load_reference_images()
    if not references:
        console.print("[red]No reference images found![/red]")
        return
    
    results = run_face_audit(references, args.threshold, args.limit)
    
    if not results:
        console.print("[yellow]No results.[/yellow]")
        return
    
    report = generate_report(results, args.threshold)
    print_summary(report)
    save_report(report)
    
    if args.html:
        generate_html_report(report)
    
    console.print("\n[bold green]Audit complete![/bold green]")


if __name__ == "__main__":
    main()
