#!/usr/bin/env python3
"""
KELLY PRODUCTION GENERATOR - Uses Trained LoRA
==============================================
Generates Kelly images using the Curious_Kelly LoRA model via Replicate.

THIS IS THE CORRECT WAY TO GENERATE KELLY.
NOT generic text-to-image. The LoRA ensures character consistency.

Usage:
    python generate_kelly_lora.py --pose idle
    python generate_kelly_lora.py --all
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime

# Load .env.local
ENV_PATH = Path(r"C:\Users\user\UI-TARS-desktop\.env.local")
if ENV_PATH.exists():
    with open(ENV_PATH) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ.setdefault(key, val)

REPLICATE_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if not REPLICATE_TOKEN:
    print("ERROR: REPLICATE_API_TOKEN not set!")
    print("Set it in .env.local or as environment variable.")
    sys.exit(1)

# LoRA URLs (the TRAINED Kelly model)
HF_LORA_URL = "https://huggingface.co/CuriousKellycom/curious-kelly-lora/resolve/main/curious_kelly.safetensors"
CIVITAI_LORA_URL = "https://civitai.com/api/download/models/2455956"

# Output directories
OUTPUT_DIR = Path(r"C:\Users\user\UI-TARS-desktop\public\kelly\poses")
PHASES_DIR = Path(r"C:\Users\user\UI-TARS-desktop\public\kelly\phases")

# LOCKED character description - matches LoRA training
KELLY = """kelly, woman with brown wavy shoulder-length hair with caramel highlights center-parted, 
hazel-brown eyes, soft natural features, light natural makeup, 
wearing soft powder blue cashmere crewneck sweater, medium wash blue jeans cuffed at ankle, 
white leather sneakers"""

# LOCKED scene description
SCENE = """pure white cyclorama photography studio, director's chair with black canvas fabric seat 
and natural warm wood frame with round finials, professional studio lighting with soft natural 
window light from upper right casting gentle diagonal shadows on light gray seamless floor, 
clean minimal background, shot on Hasselblad H6D-100c, 85mm f/2.8, shallow depth of field, 
professional fashion photography, 8K UHD"""

# 12 Core poses
POSES = {
    "idle": f"{KELLY}, seated in director's chair, relaxed natural posture, genuine warm smile, looking directly at camera, hands resting on armrests, {SCENE}",
    "thinking": f"{KELLY}, seated in director's chair, chin resting thoughtfully on right hand, elbow on armrest, looking upward and slightly to side, contemplative curious expression, {SCENE}",
    "pointing_left": f"{KELLY}, seated in director's chair, left arm extended gracefully pointing to the left, body angled slightly left, looking toward the left, encouraging helpful expression, {SCENE}",
    "pointing_right": f"{KELLY}, seated in director's chair, right arm extended gracefully pointing to the right, body angled slightly right, looking toward the right, encouraging helpful expression, {SCENE}",
    "pointing_up": f"{KELLY}, seated in director's chair, right arm raised elegantly pointing upward, head tilted back slightly, looking up, engaged interested expression, {SCENE}",
    "pointing_down": f"{KELLY}, seated in director's chair, right arm lowered gracefully pointing downward, slight forward lean, looking down, helpful guiding expression, {SCENE}",
    "encouraging": f"{KELLY}, seated in director's chair, leaning slightly forward, warm open welcoming expression, supportive smile, engaged body language, hands visible in welcoming gesture, {SCENE}",
    "hint": f"{KELLY}, seated in director's chair, playful knowing expression, index finger touching lips in secret gesture, slight mischievous smirk, head tilted playfully, {SCENE}",
    "celebrating": f"{KELLY}, seated in director's chair, both arms raised joyfully in celebration, big genuine smile, bright excited eyes, triumphant victorious pose, {SCENE}",
    "supportive": f"{KELLY}, seated in director's chair, warm empathetic caring expression, gentle head tilt showing understanding, reassuring smile, one hand on heart, {SCENE}",
    "proud": f"{KELLY}, seated in director's chair, right hand placed meaningfully on heart, satisfied accomplished smile, dignified confident posture, content proud expression, {SCENE}",
    "excited": f"{KELLY}, seated in director's chair, energetic forward-leaning posture, bright enthusiastic eyes, excited anticipating smile, hands clasped together eagerly, {SCENE}",
}

# Phase-specific poses for lessons
PHASE_POSES = {
    "hook": f"{KELLY}, seated in director's chair, warm welcoming smile, inviting pose, about to share something exciting, {SCENE}",
    "q1": f"{KELLY}, seated in director's chair, curious expression, head tilted slightly, asking first question, {SCENE}",
    "q2": f"{KELLY}, seated in director's chair, thoughtful expression, hand near chin, contemplating, {SCENE}",
    "q3": f"{KELLY}, seated in director's chair, engaged expression, leaning forward slightly, interested, {SCENE}",
    "wisdom": f"{KELLY}, seated in director's chair, serene wise smile, calm composed demeanor, sharing wisdom, {SCENE}",
}


def generate_with_lora(prompt: str, output_path: Path, aspect_ratio: str = "16:9") -> bool:
    """Generate an image using Replicate FLUX + LoRA"""
    
    print(f"  Generating with LoRA...")
    
    headers = {
        "Authorization": f"Token {REPLICATE_TOKEN}",
        "Content-Type": "application/json",
    }
    
    # Use FLUX Dev with LoRA
    payload = {
        "version": "a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
        "input": {
            "prompt": prompt,
            "hf_lora": HF_LORA_URL,
            "lora_scale": 0.85,
            "num_outputs": 1,
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
            "guidance_scale": 3.5,
            "output_quality": 100,
            "prompt_strength": 0.8,
            "num_inference_steps": 28,
            "disable_safety_checker": True,
        }
    }
    
    try:
        # Create prediction
        resp = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        prediction = resp.json()
        prediction_id = prediction["id"]
        
        print(f"  Prediction started: {prediction_id}")
        
        # Poll for completion
        for _ in range(120):  # Max 10 minutes
            time.sleep(5)
            poll = requests.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers=headers,
                timeout=30
            )
            poll.raise_for_status()
            result = poll.json()
            
            status = result["status"]
            if status == "succeeded":
                output_url = result["output"]
                if isinstance(output_url, list):
                    output_url = output_url[0]
                
                # Download image
                img_resp = requests.get(output_url, timeout=60)
                img_resp.raise_for_status()
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(img_resp.content)
                print(f"  [OK] Saved: {output_path.name}")
                return True
                
            elif status == "failed":
                print(f"  [ERROR] Generation failed: {result.get('error', 'Unknown error')}")
                return False
            
            print(f"  Status: {status}...")
        
        print("  [ERROR] Timeout waiting for generation")
        return False
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def generate_pose(pose_name: str) -> bool:
    """Generate a single pose"""
    if pose_name not in POSES:
        print(f"Unknown pose: {pose_name}")
        print(f"Available poses: {', '.join(POSES.keys())}")
        return False
    
    prompt = POSES[pose_name]
    output_path = OUTPUT_DIR / f"kelly_{pose_name}.png"
    
    print(f"\n{'='*60}")
    print(f"Generating: {pose_name}")
    print(f"{'='*60}")
    
    return generate_with_lora(prompt, output_path)


def generate_phase(lesson_num: str, phase_name: str) -> bool:
    """Generate a phase image for a lesson"""
    if phase_name not in PHASE_POSES:
        print(f"Unknown phase: {phase_name}")
        return False
    
    prompt = PHASE_POSES[phase_name]
    output_path = PHASES_DIR / lesson_num / f"{phase_name}.png"
    
    print(f"  Phase: {phase_name}")
    return generate_with_lora(prompt, output_path)


def generate_all_poses() -> dict:
    """Generate all 12 core poses"""
    results = {"success": 0, "failed": 0}
    
    print("\n" + "="*60)
    print("KELLY PRODUCTION GENERATOR - ALL POSES")
    print("Using trained LoRA: Curious_Kelly.safetensors")
    print("="*60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for pose_name in POSES:
        if generate_pose(pose_name):
            results["success"] += 1
        else:
            results["failed"] += 1
        
        # Rate limit
        time.sleep(2)
    
    print("\n" + "="*60)
    print(f"COMPLETE: {results['success']}/{len(POSES)} succeeded")
    print("="*60)
    
    return results


def generate_lesson_phases(lesson_num: str) -> dict:
    """Generate all phase images for a lesson"""
    results = {"success": 0, "failed": 0}
    
    print(f"\n{'='*60}")
    print(f"GENERATING LESSON {lesson_num} PHASES")
    print("Using trained LoRA: Curious_Kelly.safetensors")
    print(f"{'='*60}")
    
    lesson_dir = PHASES_DIR / lesson_num
    lesson_dir.mkdir(parents=True, exist_ok=True)
    
    for phase_name in PHASE_POSES:
        if generate_phase(lesson_num, phase_name):
            results["success"] += 1
        else:
            results["failed"] += 1
        time.sleep(2)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {results['success']}/{len(PHASE_POSES)} succeeded")
    print(f"{'='*60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate Kelly images with trained LoRA")
    parser.add_argument("--pose", help="Generate specific pose (idle, thinking, etc.)")
    parser.add_argument("--all", action="store_true", help="Generate all 12 poses")
    parser.add_argument("--lesson", help="Generate phases for lesson (e.g., 001)")
    args = parser.parse_args()
    
    if args.all:
        generate_all_poses()
    elif args.pose:
        generate_pose(args.pose)
    elif args.lesson:
        generate_lesson_phases(args.lesson)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python generate_kelly_lora.py --pose idle")
        print("  python generate_kelly_lora.py --all")
        print("  python generate_kelly_lora.py --lesson 001")


if __name__ == "__main__":
    main()


