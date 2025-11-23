#!/usr/bin/env python3
"""
Validate Age Consistency for Kelly Age-Progressive Images
Checks for consistency, quality issues, and generates validation report
"""

from pathlib import Path
from typing import Dict, List
import json


def scan_and_validate(renders_dir: Path) -> Dict:
    """Scan renders directory and validate image set"""
    
    ages = ["3", "9", "15", "27", "48", "82"]
    poses = ["pose1", "pose2", "pose3", "pose4"]
    formats = ["16x9", "1x1", "3x4"]
    
    results = {
        "total_expected": len(ages) * len(poses) * len(formats),
        "total_found": 0,
        "missing": [],
        "by_age": {},
        "by_pose": {},
        "by_format": {}
    }
    
    # Initialize counters
    for age in ages:
        results["by_age"][age] = {"found": 0, "missing": 0}
    for pose in poses:
        results["by_pose"][pose] = {"found": 0, "missing": 0}
    for fmt in formats:
        results["by_format"][fmt] = {"found": 0, "missing": 0}
    
    # Scan for images
    for age in ages:
        for pose in poses:
            for fmt in formats:
                filename = f"kelly_age{age}_{pose}_{fmt}.png"
                filepath = renders_dir / filename
                
                if filepath.exists():
                    results["total_found"] += 1
                    results["by_age"][age]["found"] += 1
                    results["by_pose"][pose]["found"] += 1
                    results["by_format"][fmt]["found"] += 1
                else:
                    results["missing"].append(filename)
                    results["by_age"][age]["missing"] += 1
                    results["by_pose"][pose]["missing"] += 1
                    results["by_format"][fmt]["missing"] += 1
    
    return results


def generate_report(results: Dict, output_path: Path):
    """Generate validation report"""
    
    report = f"""
# Kelly Age-Progressive Images - Validation Report

Generated: {output_path.parent.name}

## Overall Statistics

- **Total Expected**: {results['total_expected']} images
- **Total Found**: {results['total_found']} images
- **Success Rate**: {(results['total_found']/results['total_expected']*100):.1f}%
- **Missing**: {len(results['missing'])} images

## Breakdown by Age

"""
    
    for age, stats in results["by_age"].items():
        total = stats["found"] + stats["missing"]
        report += f"- **Age {age}**: {stats['found']}/{total} ({(stats['found']/total*100):.0f}%)\n"
    
    report += "\n## Breakdown by Pose\n\n"
    
    for pose, stats in results["by_pose"].items():
        total = stats["found"] + stats["missing"]
        report += f"- **{pose}**: {stats['found']}/{total} ({(stats['found']/total*100):.0f}%)\n"
    
    report += "\n## Breakdown by Format\n\n"
    
    for fmt, stats in results["by_format"].items():
        total = stats["found"] + stats["missing"]
        report += f"- **{fmt}**: {stats['found']}/{total} ({(stats['found']/total*100):.0f}%)\n"
    
    if results["missing"]:
        report += "\n## Missing Images\n\n"
        for filename in results["missing"]:
            report += f"- {filename}\n"
    
    report += "\n## Validation Checklist\n\n"
    report += "Manual review required for:\n\n"
    report += "- [ ] Kelly's identity recognizable across all ages\n"
    report += "- [ ] Aging progression looks natural\n"
    report += "- [ ] Poses consistent across age groups\n"
    report += "- [ ] Blue sweater visible in all images\n"
    report += "- [ ] Director's chair visible where expected\n"
    report += "- [ ] Warm smile consistent\n"
    report += "- [ ] Background/lighting consistent\n"
    report += "- [ ] No anatomical distortions\n"
    report += "- [ ] Aspect ratios correct\n"
    report += "- [ ] Image quality acceptable for 3D modeling reference\n"
    
    report += "\n## Next Steps\n\n"
    
    if results["total_found"] == results["total_expected"]:
        report += "✅ All images generated successfully!\n\n"
        report += "1. Review images in gallery: `review.html`\n"
        report += "2. Complete manual validation checklist\n"
        report += "3. If quality acceptable, proceed to 3D modeling\n"
        report += "4. If quality insufficient, retry with Replicate InstantID\n"
    else:
        report += f"⚠️ {len(results['missing'])} images missing!\n\n"
        report += "1. Check generation log for errors\n"
        report += "2. Retry failed images: `generate_kelly_batch_ages.ps1`\n"
        report += "3. Review successfully generated images\n"
    
    # Write report
    output_path.write_text(report, encoding='utf-8')
    print(f"\n✅ Validation report generated: {output_path}")
    print(report)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate age-progressive image generation")
    parser.add_argument("--renders-dir", type=Path, default=Path("projects/Kelly/assets/age_progressive/renders"))
    parser.add_argument("--output", type=Path, default=Path("projects/Kelly/assets/age_progressive/validation_report.md"))
    
    args = parser.parse_args()
    
    if not args.renders_dir.exists():
        print(f"⚠️  Renders directory not found: {args.renders_dir}")
        print("Run generation first: .\\scripts\\generate_kelly_batch_ages.ps1")
        return
    
    print(f"Validating images in: {args.renders_dir}")
    results = scan_and_validate(args.renders_dir)
    generate_report(results, args.output)


if __name__ == "__main__":
    main()



