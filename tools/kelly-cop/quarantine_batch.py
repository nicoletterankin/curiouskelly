#!/usr/bin/env python3
"""
KELLY COP - Batch Quarantine Script
====================================
Moves all flagged files to quarantine while preserving folder structure.
"""

import os
import shutil
import csv
from pathlib import Path
from datetime import datetime

# Paths
AUDIT_DETAILS = Path(r"C:\Users\user\UI-TARS-desktop\tools\kelly-cop\face_audit_report\face_audit_details_20251206_025028.csv")
QUARANTINE_BASE = Path(r"C:\Users\user\UI-TARS-desktop\_quarantine\kelly-imposters")
NO_MATCH_DIR = QUARANTINE_BASE / "no-match-batch-20251206"
SUSPICIOUS_DIR = QUARANTINE_BASE / "suspicious-batch-20251206"

def load_audit_results():
    """Load the audit CSV and categorize files"""
    no_match = []
    suspicious = []
    
    with open(AUDIT_DETAILS, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] == 'NO_MATCH':
                no_match.append(row['file_path'].strip('"'))
            elif row['status'] == 'SUSPICIOUS':
                suspicious.append(row['file_path'].strip('"'))
    
    return no_match, suspicious

def quarantine_file(src_path: str, quarantine_dir: Path) -> bool:
    """Move a file to quarantine, preserving relative structure"""
    src = Path(src_path)
    if not src.exists():
        print(f"  [SKIP] Not found: {src.name}")
        return False
    
    # Create relative path from UI-TARS-desktop
    try:
        rel_path = src.relative_to(r"C:\Users\user\UI-TARS-desktop")
    except ValueError:
        rel_path = Path(src.name)
    
    # Create destination maintaining folder structure
    dst = quarantine_dir / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle duplicates
    if dst.exists():
        stem = dst.stem
        suffix = dst.suffix
        counter = 1
        while dst.exists():
            dst = dst.parent / f"{stem}_{counter}{suffix}"
            counter += 1
    
    shutil.move(str(src), str(dst))
    return True

def main():
    print("=" * 60)
    print("KELLY COP - BATCH QUARANTINE")
    print("=" * 60)
    print()
    
    # Load results
    print("Loading audit results...")
    no_match, suspicious = load_audit_results()
    print(f"  NO_MATCH files: {len(no_match)}")
    print(f"  SUSPICIOUS files: {len(suspicious)}")
    print()
    
    # Quarantine NO_MATCH files
    print(f"Quarantining {len(no_match)} NO_MATCH files...")
    moved_nm = 0
    for path in no_match:
        if quarantine_file(path, NO_MATCH_DIR):
            moved_nm += 1
    print(f"  Moved: {moved_nm}/{len(no_match)}")
    print()
    
    # Quarantine SUSPICIOUS files
    print(f"Quarantining {len(suspicious)} SUSPICIOUS files...")
    moved_sus = 0
    for path in suspicious:
        if quarantine_file(path, SUSPICIOUS_DIR):
            moved_sus += 1
    print(f"  Moved: {moved_sus}/{len(suspicious)}")
    print()
    
    # Summary
    print("=" * 60)
    print("QUARANTINE COMPLETE")
    print("=" * 60)
    print(f"  NO_MATCH moved: {moved_nm}")
    print(f"  SUSPICIOUS moved: {moved_sus}")
    print(f"  Total quarantined: {moved_nm + moved_sus}")
    print()
    print(f"Quarantine location: {QUARANTINE_BASE}")
    print()
    
    # Create manifest
    manifest_path = QUARANTINE_BASE / f"QUARANTINE_BATCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(manifest_path, 'w') as f:
        f.write(f"# Quarantine Batch - {datetime.now().isoformat()}\n\n")
        f.write(f"## Summary\n")
        f.write(f"- NO_MATCH files quarantined: {moved_nm}\n")
        f.write(f"- SUSPICIOUS files quarantined: {moved_sus}\n")
        f.write(f"- Total: {moved_nm + moved_sus}\n\n")
        f.write(f"## Reason\n")
        f.write(f"Face recognition audit determined these images do not match canonical 'Our Kelly' reference.\n\n")
        f.write(f"## Next Steps\n")
        f.write(f"1. Review quarantined files if needed\n")
        f.write(f"2. Regenerate all lesson assets with correct Kelly reference\n")
        f.write(f"3. Delete quarantine after verification\n")
    
    print(f"Manifest saved: {manifest_path}")

if __name__ == "__main__":
    main()

