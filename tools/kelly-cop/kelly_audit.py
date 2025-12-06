#!/usr/bin/env python3
"""
🚔 KELLY COP - Visual Identity Audit Tool
==========================================
Scans all Kelly images and verifies they match "Our Kelly" using perceptual hashing.

Usage:
    python kelly_audit.py [--scan-only] [--quarantine] [--report]
    
Dependencies:
    pip install Pillow imagehash rich
"""

import os
import sys
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from PIL import Image
    import imagehash
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.panel import Panel
    from rich import print as rprint
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install Pillow imagehash rich")
    sys.exit(1)

console = Console()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Canonical reference folder
CANONICAL_REF_PATH = Path(r"C:\iLearnStudio\projects\Kelly\Ref\Best Character Reference")

# Folders to scan for Kelly images
SCAN_FOLDERS = [
    Path(r"C:\Users\user\UI-TARS-desktop\public\kelly"),
    Path(r"C:\Users\user\UI-TARS-desktop\generated-poses-final"),
    Path(r"C:\Users\user\UI-TARS-desktop\generated-poses-production"),
    Path(r"C:\Users\user\UI-TARS-desktop\generated-poses-presenter"),
    Path(r"C:\Users\user\UI-TARS-desktop\daily-lesson-marketing\public\assets\kelly"),
]

# Additional root-level files to check
ROOT_KELLY_FILES = [
    Path(r"C:\Users\user\UI-TARS-desktop\kelly-lora-test.png"),
    Path(r"C:\Users\user\UI-TARS-desktop\test_kelly_character_consistent.png"),
]

# Quarantine location
QUARANTINE_PATH = Path(r"C:\Users\user\UI-TARS-desktop\_quarantine\kelly-imposters")

# Output report location
REPORT_PATH = Path(r"C:\Users\user\UI-TARS-desktop\tools\kelly-cop\audit_report")

# Image extensions to scan
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}

# Thresholds for perceptual hash comparison (lower = more similar)
# These are Hamming distances for 16x16 hashes (0-256 range)
# Note: pHash compares ENTIRE image, not just faces
# Different poses/backgrounds cause high distances even for same person
THRESHOLD_GOOD = 90       # Same composition/pose as reference
THRESHOLD_SUSPICIOUS = 110  # Different pose but plausibly same person
THRESHOLD_IMPOSTER = 130   # Very different - needs manual review


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ImageAuditResult:
    """Result of auditing a single image"""
    file_path: str
    file_name: str
    folder: str
    phash: str
    avg_hash: str
    dhash: str
    min_distance: int  # Minimum hamming distance to any reference
    closest_reference: str  # Which reference image it's closest to
    status: str  # GOOD, SUSPICIOUS, IMPOSTER
    confidence: float  # 0-100% confidence it's Our Kelly
    file_size_kb: float
    dimensions: Tuple[int, int]
    error: Optional[str] = None


@dataclass
class FolderSummary:
    """Summary of audit results for a folder"""
    folder_path: str
    total_images: int = 0
    good_count: int = 0
    suspicious_count: int = 0
    imposter_count: int = 0
    error_count: int = 0
    avg_confidence: float = 0.0


@dataclass
class AuditReport:
    """Complete audit report"""
    timestamp: str
    canonical_refs_used: List[str]
    total_images_scanned: int = 0
    total_good: int = 0
    total_suspicious: int = 0
    total_imposters: int = 0
    total_errors: int = 0
    folder_summaries: List[FolderSummary] = field(default_factory=list)
    imposter_files: List[str] = field(default_factory=list)
    suspicious_files: List[str] = field(default_factory=list)


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_image(path: Path) -> Optional[Image.Image]:
    """Load an image, handling errors gracefully"""
    try:
        img = Image.open(path)
        # Convert to RGB if necessary (handles RGBA, P mode, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        console.print(f"[red]Error loading {path}: {e}[/red]")
        return None


def compute_hashes(img: Image.Image) -> Tuple[str, str, str]:
    """Compute multiple perceptual hashes for an image"""
    phash = str(imagehash.phash(img, hash_size=16))
    avg_hash = str(imagehash.average_hash(img, hash_size=16))
    dhash = str(imagehash.dhash(img, hash_size=16))
    return phash, avg_hash, dhash


def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculate Hamming distance between two hash strings"""
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    return h1 - h2


def load_canonical_references() -> Dict[str, Dict[str, str]]:
    """Load and compute hashes for all canonical reference images"""
    references = {}
    
    if not CANONICAL_REF_PATH.exists():
        console.print(f"[red]Canonical reference path not found: {CANONICAL_REF_PATH}[/red]")
        return references
    
    console.print(f"\n[cyan]📂 Loading canonical references from:[/cyan] {CANONICAL_REF_PATH}")
    
    for file in CANONICAL_REF_PATH.iterdir():
        if file.suffix.lower() in IMAGE_EXTENSIONS:
            img = load_image(file)
            if img:
                phash, avg_hash, dhash = compute_hashes(img)
                references[file.name] = {
                    'path': str(file),
                    'phash': phash,
                    'avg_hash': avg_hash,
                    'dhash': dhash
                }
                console.print(f"  [green]✓[/green] {file.name}")
    
    console.print(f"[green]Loaded {len(references)} canonical references[/green]\n")
    return references


def compare_to_references(
    phash: str, 
    avg_hash: str, 
    dhash: str, 
    references: Dict[str, Dict[str, str]]
) -> Tuple[int, str, float]:
    """
    Compare image hashes to all references.
    Returns: (min_distance, closest_reference_name, confidence_score)
    """
    min_distance = float('inf')
    closest_ref = "none"
    
    for ref_name, ref_hashes in references.items():
        # Use weighted combination of different hash types
        # pHash is most reliable for facial recognition
        p_dist = hamming_distance(phash, ref_hashes['phash'])
        a_dist = hamming_distance(avg_hash, ref_hashes['avg_hash'])
        d_dist = hamming_distance(dhash, ref_hashes['dhash'])
        
        # Weighted average (pHash weighted higher)
        combined_dist = (p_dist * 0.5) + (a_dist * 0.25) + (d_dist * 0.25)
        
        if combined_dist < min_distance:
            min_distance = combined_dist
            closest_ref = ref_name
    
    # Convert distance to confidence (0-100%)
    # Lower distance = higher confidence
    # At threshold_good (90), confidence should be ~85%
    # At threshold_imposter (130), confidence should be ~20%
    # Scale: distance 50 = 100%, distance 150 = 0%
    confidence = max(0, min(100, 100 - ((min_distance - 50) * 1.0)))
    
    return int(min_distance), closest_ref, confidence


def classify_status(distance: int) -> str:
    """Classify image status based on distance to references"""
    if distance <= THRESHOLD_GOOD:
        return "GOOD"
    elif distance <= THRESHOLD_SUSPICIOUS:
        return "SUSPICIOUS"
    else:
        return "IMPOSTER"


def audit_image(
    file_path: Path, 
    references: Dict[str, Dict[str, str]]
) -> ImageAuditResult:
    """Audit a single image against canonical references"""
    try:
        img = load_image(file_path)
        if img is None:
            return ImageAuditResult(
                file_path=str(file_path),
                file_name=file_path.name,
                folder=str(file_path.parent),
                phash="", avg_hash="", dhash="",
                min_distance=999,
                closest_reference="error",
                status="ERROR",
                confidence=0,
                file_size_kb=0,
                dimensions=(0, 0),
                error="Failed to load image"
            )
        
        # Compute hashes
        phash, avg_hash, dhash = compute_hashes(img)
        
        # Compare to references
        min_distance, closest_ref, confidence = compare_to_references(
            phash, avg_hash, dhash, references
        )
        
        # Classify
        status = classify_status(min_distance)
        
        # Get file metadata
        file_size_kb = file_path.stat().st_size / 1024
        dimensions = img.size
        
        return ImageAuditResult(
            file_path=str(file_path),
            file_name=file_path.name,
            folder=str(file_path.parent),
            phash=phash,
            avg_hash=avg_hash,
            dhash=dhash,
            min_distance=min_distance,
            closest_reference=closest_ref,
            status=status,
            confidence=confidence,
            file_size_kb=round(file_size_kb, 2),
            dimensions=dimensions
        )
        
    except Exception as e:
        return ImageAuditResult(
            file_path=str(file_path),
            file_name=file_path.name,
            folder=str(file_path.parent),
            phash="", avg_hash="", dhash="",
            min_distance=999,
            closest_reference="error",
            status="ERROR",
            confidence=0,
            file_size_kb=0,
            dimensions=(0, 0),
            error=str(e)
        )


def collect_image_files(folders: List[Path], root_files: List[Path]) -> List[Path]:
    """Collect all image files to audit"""
    files = []
    
    # Add root-level files
    for f in root_files:
        if f.exists() and f.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(f)
    
    # Recursively scan folders
    for folder in folders:
        if folder.exists():
            for ext in IMAGE_EXTENSIONS:
                files.extend(folder.rglob(f"*{ext}"))
    
    return files


def run_audit(references: Dict[str, Dict[str, str]], max_workers: int = 8) -> List[ImageAuditResult]:
    """Run the full audit on all Kelly images"""
    
    # Collect files to scan
    console.print("[cyan]📂 Collecting image files to audit...[/cyan]")
    files = collect_image_files(SCAN_FOLDERS, ROOT_KELLY_FILES)
    console.print(f"[green]Found {len(files)} images to audit[/green]\n")
    
    if not files:
        console.print("[yellow]No images found to audit![/yellow]")
        return []
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("🔍 Auditing images...", total=len(files))
        
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(audit_image, f, references): f for f in files}
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                progress.update(task, advance=1)
    
    return results


def generate_report(results: List[ImageAuditResult], references: Dict[str, Dict[str, str]]) -> AuditReport:
    """Generate audit report from results"""
    
    report = AuditReport(
        timestamp=datetime.now().isoformat(),
        canonical_refs_used=list(references.keys()),
        total_images_scanned=len(results)
    )
    
    # Group by folder
    folder_results: Dict[str, List[ImageAuditResult]] = {}
    for r in results:
        folder = r.folder
        if folder not in folder_results:
            folder_results[folder] = []
        folder_results[folder].append(r)
    
    # Process each folder
    for folder, folder_items in folder_results.items():
        summary = FolderSummary(folder_path=folder, total_images=len(folder_items))
        
        confidences = []
        for item in folder_items:
            if item.status == "GOOD":
                summary.good_count += 1
                report.total_good += 1
            elif item.status == "SUSPICIOUS":
                summary.suspicious_count += 1
                report.total_suspicious += 1
                report.suspicious_files.append(item.file_path)
            elif item.status == "IMPOSTER":
                summary.imposter_count += 1
                report.total_imposters += 1
                report.imposter_files.append(item.file_path)
            else:  # ERROR
                summary.error_count += 1
                report.total_errors += 1
            
            if item.confidence > 0:
                confidences.append(item.confidence)
        
        if confidences:
            summary.avg_confidence = round(sum(confidences) / len(confidences), 1)
        
        report.folder_summaries.append(summary)
    
    return report


def print_summary(report: AuditReport):
    """Print a summary table of the audit results"""
    
    console.print("\n")
    console.print(Panel.fit(
        "🚔 KELLY COP AUDIT COMPLETE",
        style="bold cyan"
    ))
    
    # Overall summary
    console.print(f"\n[bold]📊 OVERALL SUMMARY[/bold]")
    console.print(f"  Total images scanned: [cyan]{report.total_images_scanned}[/cyan]")
    console.print(f"  ✅ Good (Our Kelly):   [green]{report.total_good}[/green]")
    console.print(f"  ⚠️  Suspicious:        [yellow]{report.total_suspicious}[/yellow]")
    console.print(f"  🚨 Imposters:          [red]{report.total_imposters}[/red]")
    console.print(f"  ❌ Errors:             [dim]{report.total_errors}[/dim]")
    
    # Folder breakdown table
    table = Table(title="\n📁 Folder Breakdown", show_header=True, header_style="bold magenta")
    table.add_column("Folder", style="dim", max_width=50)
    table.add_column("Total", justify="right")
    table.add_column("Good", justify="right", style="green")
    table.add_column("Suspicious", justify="right", style="yellow")
    table.add_column("Imposter", justify="right", style="red")
    table.add_column("Confidence", justify="right")
    
    for summary in sorted(report.folder_summaries, key=lambda x: x.imposter_count, reverse=True):
        # Shorten folder path for display
        folder_display = summary.folder_path
        if len(folder_display) > 50:
            folder_display = "..." + folder_display[-47:]
        
        conf_style = "green" if summary.avg_confidence >= 70 else "yellow" if summary.avg_confidence >= 40 else "red"
        
        table.add_row(
            folder_display,
            str(summary.total_images),
            str(summary.good_count),
            str(summary.suspicious_count),
            str(summary.imposter_count),
            f"[{conf_style}]{summary.avg_confidence}%[/{conf_style}]"
        )
    
    console.print(table)
    
    # List imposter files
    if report.imposter_files:
        console.print(f"\n[bold red]🚨 IMPOSTER FILES ({len(report.imposter_files)}):[/bold red]")
        for f in report.imposter_files[:20]:  # Show first 20
            console.print(f"  [red]• {f}[/red]")
        if len(report.imposter_files) > 20:
            console.print(f"  [dim]... and {len(report.imposter_files) - 20} more[/dim]")
    
    # List suspicious files (first few)
    if report.suspicious_files:
        console.print(f"\n[bold yellow]⚠️ SUSPICIOUS FILES ({len(report.suspicious_files)}):[/bold yellow]")
        for f in report.suspicious_files[:10]:  # Show first 10
            console.print(f"  [yellow]• {f}[/yellow]")
        if len(report.suspicious_files) > 10:
            console.print(f"  [dim]... and {len(report.suspicious_files) - 10} more[/dim]")


def save_report(report: AuditReport, results: List[ImageAuditResult]):
    """Save detailed report to files"""
    
    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary JSON
    summary_file = REPORT_PATH / f"audit_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(asdict(report), f, indent=2)
    console.print(f"\n[green]📄 Summary saved to:[/green] {summary_file}")
    
    # Save detailed results CSV
    csv_file = REPORT_PATH / f"audit_details_{timestamp}.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("file_path,file_name,folder,status,confidence,min_distance,closest_reference,dimensions,file_size_kb,phash\n")
        for r in results:
            dims = f"{r.dimensions[0]}x{r.dimensions[1]}"
            f.write(f'"{r.file_path}","{r.file_name}","{r.folder}",{r.status},{r.confidence},{r.min_distance},"{r.closest_reference}",{dims},{r.file_size_kb},{r.phash}\n')
    console.print(f"[green]📊 Details saved to:[/green] {csv_file}")
    
    # Save imposter list for easy action
    if report.imposter_files:
        imposter_file = REPORT_PATH / f"imposters_{timestamp}.txt"
        with open(imposter_file, 'w') as f:
            for path in report.imposter_files:
                f.write(path + "\n")
        console.print(f"[red]🚨 Imposter list saved to:[/red] {imposter_file}")
    
    return summary_file, csv_file


def quarantine_imposters(report: AuditReport):
    """Move imposter files to quarantine folder"""
    
    if not report.imposter_files:
        console.print("[green]No imposters to quarantine![/green]")
        return
    
    console.print(f"\n[yellow]⚠️ About to quarantine {len(report.imposter_files)} imposter files[/yellow]")
    confirm = input("Proceed? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        console.print("[dim]Quarantine cancelled[/dim]")
        return
    
    QUARANTINE_PATH.mkdir(parents=True, exist_ok=True)
    moved = 0
    
    for file_path in report.imposter_files:
        src = Path(file_path)
        if src.exists():
            # Preserve folder structure in quarantine
            rel_path = src.name  # Just use filename to avoid deep nesting
            dst = QUARANTINE_PATH / rel_path
            
            # Handle duplicates
            if dst.exists():
                stem = dst.stem
                suffix = dst.suffix
                counter = 1
                while dst.exists():
                    dst = QUARANTINE_PATH / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            shutil.move(str(src), str(dst))
            moved += 1
    
    console.print(f"[green]✓ Moved {moved} files to {QUARANTINE_PATH}[/green]")


# =============================================================================
# HTML REPORT GENERATOR
# =============================================================================

def generate_html_report(results: List[ImageAuditResult], references: Dict[str, Dict[str, str]], report: AuditReport):
    """Generate visual HTML report with image comparisons"""
    
    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_file = REPORT_PATH / f"visual_report_{timestamp}.html"
    
    # Group results by status
    imposters = [r for r in results if r.status == "IMPOSTER"]
    suspicious = [r for r in results if r.status == "SUSPICIOUS"]
    good = [r for r in results if r.status == "GOOD"][:50]  # Limit good to 50 for performance
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>🚔 Kelly Cop Audit Report - {timestamp}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e; 
            color: #eee; 
            margin: 0; 
            padding: 20px;
        }}
        h1 {{ color: #00d9ff; text-align: center; }}
        h2 {{ color: #ffd700; border-bottom: 2px solid #ffd700; padding-bottom: 10px; }}
        .stats {{ 
            display: flex; 
            justify-content: center; 
            gap: 30px; 
            margin: 30px 0;
            flex-wrap: wrap;
        }}
        .stat-box {{ 
            background: #16213e; 
            padding: 20px 40px; 
            border-radius: 10px;
            text-align: center;
        }}
        .stat-box.good {{ border: 2px solid #00ff88; }}
        .stat-box.suspicious {{ border: 2px solid #ffaa00; }}
        .stat-box.imposter {{ border: 2px solid #ff4444; }}
        .stat-number {{ font-size: 48px; font-weight: bold; }}
        .stat-label {{ font-size: 14px; opacity: 0.8; }}
        .grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); 
            gap: 20px;
            margin: 20px 0;
        }}
        .card {{
            background: #16213e;
            border-radius: 10px;
            overflow: hidden;
        }}
        .card.imposter {{ border: 3px solid #ff4444; }}
        .card.suspicious {{ border: 3px solid #ffaa00; }}
        .card.good {{ border: 3px solid #00ff88; }}
        .card img {{
            width: 100%;
            height: 200px;
            object-fit: cover;
            background: #0f0f23;
        }}
        .card-info {{
            padding: 15px;
        }}
        .card-title {{
            font-weight: bold;
            word-break: break-all;
            font-size: 12px;
            margin-bottom: 10px;
        }}
        .card-meta {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            opacity: 0.8;
        }}
        .confidence {{
            font-size: 24px;
            font-weight: bold;
        }}
        .confidence.high {{ color: #00ff88; }}
        .confidence.medium {{ color: #ffaa00; }}
        .confidence.low {{ color: #ff4444; }}
        .reference-section {{
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin: 30px 0;
        }}
        .reference-grid {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            justify-content: center;
        }}
        .reference-grid img {{
            width: 150px;
            height: 150px;
            object-fit: cover;
            border-radius: 10px;
            border: 2px solid #00d9ff;
        }}
        .section {{ margin: 40px 0; }}
    </style>
</head>
<body>
    <h1>🚔 KELLY COP AUDIT REPORT</h1>
    <p style="text-align: center; opacity: 0.7;">Generated: {timestamp}</p>
    
    <div class="stats">
        <div class="stat-box">
            <div class="stat-number" style="color: #00d9ff;">{report.total_images_scanned}</div>
            <div class="stat-label">Total Scanned</div>
        </div>
        <div class="stat-box good">
            <div class="stat-number" style="color: #00ff88;">{report.total_good}</div>
            <div class="stat-label">✅ Good</div>
        </div>
        <div class="stat-box suspicious">
            <div class="stat-number" style="color: #ffaa00;">{report.total_suspicious}</div>
            <div class="stat-label">⚠️ Suspicious</div>
        </div>
        <div class="stat-box imposter">
            <div class="stat-number" style="color: #ff4444;">{report.total_imposters}</div>
            <div class="stat-label">🚨 Imposters</div>
        </div>
    </div>
    
    <div class="reference-section">
        <h2 style="margin-top: 0;">📸 Canonical "Our Kelly" References</h2>
        <div class="reference-grid">
"""
    
    # Add reference images
    for ref_name, ref_data in references.items():
        ref_path = ref_data['path'].replace('\\', '/')
        html += f'            <img src="file:///{ref_path}" alt="{ref_name}" title="{ref_name}">\n'
    
    html += """        </div>
    </div>
"""
    
    # Imposter section
    if imposters:
        html += f"""
    <div class="section">
        <h2>🚨 IMPOSTERS ({len(imposters)} files)</h2>
        <div class="grid">
"""
        for r in imposters:
            conf_class = "low"
            file_path = r.file_path.replace('\\', '/')
            html += f"""
            <div class="card imposter">
                <img src="file:///{file_path}" alt="{r.file_name}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22300%22 height=%22200%22><rect fill=%22%23333%22 width=%22300%22 height=%22200%22/><text fill=%22%23999%22 x=%22150%22 y=%22100%22 text-anchor=%22middle%22>Image not found</text></svg>'">
                <div class="card-info">
                    <div class="card-title">{r.file_name}</div>
                    <div class="card-meta">
                        <span class="confidence {conf_class}">{r.confidence}%</span>
                        <span>Distance: {r.min_distance}</span>
                    </div>
                </div>
            </div>
"""
        html += """        </div>
    </div>
"""
    
    # Suspicious section
    if suspicious:
        html += f"""
    <div class="section">
        <h2>⚠️ SUSPICIOUS ({len(suspicious)} files)</h2>
        <div class="grid">
"""
        for r in suspicious[:30]:  # Limit for performance
            conf_class = "medium" if r.confidence >= 40 else "low"
            file_path = r.file_path.replace('\\', '/')
            html += f"""
            <div class="card suspicious">
                <img src="file:///{file_path}" alt="{r.file_name}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22300%22 height=%22200%22><rect fill=%22%23333%22 width=%22300%22 height=%22200%22/><text fill=%22%23999%22 x=%22150%22 y=%22100%22 text-anchor=%22middle%22>Image not found</text></svg>'">
                <div class="card-info">
                    <div class="card-title">{r.file_name}</div>
                    <div class="card-meta">
                        <span class="confidence {conf_class}">{r.confidence}%</span>
                        <span>Distance: {r.min_distance}</span>
                    </div>
                </div>
            </div>
"""
        html += """        </div>
    </div>
"""
    
    # Good section (sample)
    if good:
        html += f"""
    <div class="section">
        <h2>✅ GOOD (showing {len(good)} of {report.total_good})</h2>
        <div class="grid">
"""
        for r in good:
            conf_class = "high" if r.confidence >= 70 else "medium"
            file_path = r.file_path.replace('\\', '/')
            html += f"""
            <div class="card good">
                <img src="file:///{file_path}" alt="{r.file_name}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22300%22 height=%22200%22><rect fill=%22%23333%22 width=%22300%22 height=%22200%22/><text fill=%22%23999%22 x=%22150%22 y=%22100%22 text-anchor=%22middle%22>Image not found</text></svg>'">
                <div class="card-info">
                    <div class="card-title">{r.file_name}</div>
                    <div class="card-meta">
                        <span class="confidence {conf_class}">{r.confidence}%</span>
                        <span>Distance: {r.min_distance}</span>
                    </div>
                </div>
            </div>
"""
        html += """        </div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    console.print(f"[green]🌐 Visual report saved to:[/green] {html_file}")
    return html_file


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="🚔 Kelly Cop - Visual Identity Audit Tool")
    parser.add_argument('--scan-only', action='store_true', help="Only scan, don't generate reports")
    parser.add_argument('--quarantine', action='store_true', help="Move imposters to quarantine folder")
    parser.add_argument('--html', action='store_true', help="Generate HTML visual report")
    parser.add_argument('--workers', type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]🚔 KELLY COP[/bold cyan]\n"
        "[dim]Visual Identity Audit Tool[/dim]",
        border_style="cyan"
    ))
    
    # Load canonical references
    references = load_canonical_references()
    if not references:
        console.print("[red]No canonical references found. Cannot proceed.[/red]")
        return
    
    # Run audit
    results = run_audit(references, max_workers=args.workers)
    
    if not results:
        console.print("[yellow]No results to report.[/yellow]")
        return
    
    # Generate report
    report = generate_report(results, references)
    
    # Print summary
    print_summary(report)
    
    # Save reports
    if not args.scan_only:
        save_report(report, results)
        
        # Generate HTML report if requested
        if args.html:
            generate_html_report(results, references, report)
    
    # Quarantine if requested
    if args.quarantine:
        quarantine_imposters(report)
    
    console.print("\n[bold green]✅ Audit complete![/bold green]")


if __name__ == "__main__":
    main()

