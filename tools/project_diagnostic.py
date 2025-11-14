#!/usr/bin/env python3
"""
Project Diagnostic Tool
Assesses current state of Curious Kelly project and generates a status report.
Helps both user and AI assistant understand what's working and what needs attention.
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class ProjectDiagnostic:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.status = {
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "issues": [],
            "recommendations": []
        }
    
    def check_file_exists(self, *path_parts) -> bool:
        """Check if a file exists relative to project root."""
        return (self.project_root / Path(*path_parts)).exists()
    
    def check_dir_exists(self, *path_parts) -> bool:
        """Check if a directory exists relative to project root."""
        return (self.project_root / Path(*path_parts)).is_dir()
    
    def read_file_content(self, *path_parts, limit: int = 50) -> Optional[str]:
        """Read file content (first N lines)."""
        file_path = self.project_root / Path(*path_parts)
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return ''.join(f.readlines()[:limit])
            except Exception as e:
                return f"Error reading file: {e}"
        return None
    
    def check_backend(self) -> Dict:
        """Check backend infrastructure status."""
        status = {
            "exists": False,
            "deployed": False,
            "env_configured": False,
            "endpoints": []
        }
        
        backend_paths = [
            "curious-kellly/backend",
            "api",
            "services"
        ]
        
        for path in backend_paths:
            if self.check_dir_exists(path):
                status["exists"] = True
                break
        
        # Check for .env files
        env_files = [
            "curious-kellly/backend/.env",
            ".env",
            "curious-kellly/backend/.env.example"
        ]
        
        for env_file in env_files:
            if self.check_file_exists(env_file):
                status["env_configured"] = True
                break
        
        # Check deployment configs
        deployment_indicators = [
            "deployment/setup-cloud.sh",
            "deployment/vercel.json",
            "wrangler.toml"
        ]
        
        for indicator in deployment_indicators:
            if self.check_file_exists(indicator):
                status["deployed"] = True
                break
        
        return status
    
    def check_mobile(self) -> Dict:
        """Check mobile app status."""
        status = {
            "flutter_exists": False,
            "unity_exists": False,
            "bridge_exists": False,
            "platforms": []
        }
        
        flutter_paths = [
            "curious-kellly/mobile",
            "digital-kelly/apps/kelly_app_flutter"
        ]
        
        for path in flutter_paths:
            if self.check_dir_exists(path):
                status["flutter_exists"] = True
                if self.check_file_exists(path, "pubspec.yaml"):
                    status["platforms"].append("Flutter")
                break
        
        unity_paths = [
            "digital-kelly/engines/kelly_unity_player",
            "iLearnStudio"
        ]
        
        for path in unity_paths:
            if self.check_dir_exists(path):
                status["unity_exists"] = True
                break
        
        bridge_files = [
            "digital-kelly/apps/kelly_app_flutter/lib/bridge/unity_view.dart",
            "curious-kellly/mobile/lib/bridge/unity_view.dart"
        ]
        
        for bridge_file in bridge_files:
            if self.check_file_exists(bridge_file):
                status["bridge_exists"] = True
                break
        
        return status
    
    def check_audio_pipeline(self) -> Dict:
        """Check audio generation pipeline."""
        status = {
            "elevenlabs_integrated": False,
            "audio2face_exists": False,
            "scripts_available": False,
            "sample_audio_exists": False
        }
        
        # Check ElevenLabs integration
        scripts_to_check = [
            "generate_lesson_audio_for_iclone.py",
            "curious-kellly/backend/scripts/generate_audio.py"
        ]
        
        for script in scripts_to_check:
            if self.check_file_exists(script):
                status["scripts_available"] = True
                content = self.read_file_content(script)
                if content and "elevenlabs" in content.lower():
                    status["elevenlabs_integrated"] = True
                break
        
        # Check Audio2Face
        a2f_paths = [
            "kelly_audio2face",
            "Audio2Face-3D-Samples"
        ]
        
        for path in a2f_paths:
            if self.check_dir_exists(path):
                status["audio2face_exists"] = True
                break
        
        # Check for sample audio
        audio_dirs = [
            "assets/audio",
            "lessons/audio",
            "curious-kellly/content/audio"
        ]
        
        for audio_dir in audio_dirs:
            if self.check_dir_exists(audio_dir):
                # Check if directory has files
                audio_path = self.project_root / audio_dir
                if any(audio_path.iterdir()):
                    status["sample_audio_exists"] = True
                    break
        
        return status
    
    def check_content(self) -> Dict:
        """Check content/lessons status."""
        status = {
            "lessons_exist": False,
            "lessons_count": 0,
            "phasedna_schema_exists": False,
            "multilingual": False,
            "topics_defined": False
        }
        
        # Check lessons directory
        lesson_paths = [
            "lessons",
            "curious-kellly/content/lessons",
            "content/lessons"
        ]
        
        for lesson_path in lesson_paths:
            if self.check_dir_exists(lesson_path):
                status["lessons_exist"] = True
                lesson_dir = self.project_root / lesson_path
                json_files = list(lesson_dir.glob("*.json"))
                status["lessons_count"] = len(json_files)
                
                # Check for multilingual content
                for json_file in json_files[:3]:  # Sample first 3
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # Check for language keys
                            if isinstance(data, dict):
                                if any(key in str(data).lower() for key in ['es', 'fr', 'spanish', 'french']):
                                    status["multilingual"] = True
                    except:
                        pass
                break
        
        # Check PhaseDNA schema
        schema_files = [
            "lesson-player/lesson-dna-schema.json",
            "curious-kellly/content/schemas/phasedna.json"
        ]
        
        for schema_file in schema_files:
            if self.check_file_exists(schema_file):
                status["phasedna_schema_exists"] = True
                break
        
        # Check topics
        topic_files = [
            "topics/30-universal-topics.json",
            "content/topics.json"
        ]
        
        for topic_file in topic_files:
            if self.check_file_exists(topic_file):
                status["topics_defined"] = True
                break
        
        return status
    
    def check_documentation(self) -> Dict:
        """Check key documentation status."""
        status = {
            "key_docs_exist": False,
            "docs_found": []
        }
        
        key_docs = [
            "CLAUDE.md",
            "START_HERE.md",
            "CURIOUS_KELLLY_EXECUTION_PLAN.md",
            "TECHNICAL_ALIGNMENT_MATRIX.md",
            "BUILD_PLAN.md"
        ]
        
        for doc in key_docs:
            if self.check_file_exists(doc):
                status["docs_found"].append(doc)
        
        status["key_docs_exist"] = len(status["docs_found"]) >= 3
        
        return status
    
    def check_environment(self) -> Dict:
        """Check development environment setup."""
        status = {
            "python_available": False,
            "node_available": False,
            "git_configured": False,
            "setup_scripts_exist": False
        }
        
        # Check Python
        try:
            result = subprocess.run(
                ["python", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                status["python_available"] = True
        except:
            pass
        
        # Check Node.js
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                status["node_available"] = True
        except:
            pass
        
        # Check Git
        try:
            result = subprocess.run(
                ["git", "status"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=self.project_root
            )
            if result.returncode == 0:
                status["git_configured"] = True
        except:
            pass
        
        # Check setup scripts
        setup_scripts = [
            "setup_local.ps1",
            "verify_installation.py",
            "setup-env.ps1"
        ]
        
        for script in setup_scripts:
            if self.check_file_exists(script):
                status["setup_scripts_exist"] = True
                break
        
        return status
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on diagnostic results."""
        recommendations = []
        
        backend = self.status["components"].get("backend", {})
        mobile = self.status["components"].get("mobile", {})
        audio = self.status["components"].get("audio", {})
        content = self.status["components"].get("content", {})
        env = self.status["components"].get("environment", {})
        
        if not backend.get("exists"):
            recommendations.append("🔴 Backend infrastructure not found. Consider setting up backend service.")
        
        if backend.get("exists") and not backend.get("env_configured"):
            recommendations.append("⚠️ Backend exists but .env not configured. Run setup scripts.")
        
        if not mobile.get("flutter_exists"):
            recommendations.append("🔴 Flutter mobile app not found. Check digital-kelly/apps/ or curious-kellly/mobile/")
        
        if mobile.get("flutter_exists") and not mobile.get("bridge_exists"):
            recommendations.append("⚠️ Flutter app exists but Unity bridge missing. Critical for avatar integration.")
        
        if not audio.get("elevenlabs_integrated"):
            recommendations.append("🔴 ElevenLabs integration not found. Required for voice synthesis.")
        
        if not content.get("lessons_exist"):
            recommendations.append("🔴 No lessons found. Create at least one lesson in lessons/ directory.")
        
        if content.get("lessons_exist") and not content.get("multilingual"):
            recommendations.append("⚠️ Lessons exist but multilingual support (ES/FR) not detected. Required per CLAUDE.md.")
        
        if not env.get("python_available"):
            recommendations.append("⚠️ Python not available. Required for audio generation scripts.")
        
        if not env.get("node_available"):
            recommendations.append("⚠️ Node.js not available. Required for backend services.")
        
        return recommendations
    
    def run_diagnostic(self) -> Dict:
        """Run full diagnostic."""
        print("🔍 Running project diagnostic...")
        
        self.status["components"] = {
            "backend": self.check_backend(),
            "mobile": self.check_mobile(),
            "audio": self.check_audio_pipeline(),
            "content": self.check_content(),
            "documentation": self.check_documentation(),
            "environment": self.check_environment()
        }
        
        self.status["recommendations"] = self.generate_recommendations()
        
        return self.status
    
    def print_report(self):
        """Print formatted diagnostic report."""
        print("\n" + "="*80)
        print("CURIOUS KELLY PROJECT DIAGNOSTIC REPORT")
        print("="*80)
        print(f"Generated: {self.status['timestamp']}\n")
        
        for component_name, component_status in self.status["components"].items():
            print(f"\n📦 {component_name.upper()}")
            print("-" * 80)
            for key, value in component_status.items():
                if isinstance(value, bool):
                    icon = "✅" if value else "❌"
                    print(f"  {icon} {key}: {value}")
                elif isinstance(value, list):
                    print(f"  📋 {key}: {', '.join(value) if value else 'None'}")
                else:
                    print(f"  ℹ️  {key}: {value}")
        
        if self.status["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS")
            print("-" * 80)
            for rec in self.status["recommendations"]:
                print(f"  {rec}")
        
        print("\n" + "="*80)
    
    def save_report(self, output_file: str = "project_diagnostic_report.json"):
        """Save diagnostic report to JSON file."""
        output_path = self.project_root / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.status, f, indent=2)
        print(f"\n💾 Report saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Curious Kelly Project Diagnostic Tool")
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="project_diagnostic_report.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save JSON report"
    )
    
    args = parser.parse_args()
    
    diagnostic = ProjectDiagnostic(args.root)
    diagnostic.run_diagnostic()
    diagnostic.print_report()
    
    if not args.no_save:
        diagnostic.save_report(args.output)


if __name__ == "__main__":
    main()
