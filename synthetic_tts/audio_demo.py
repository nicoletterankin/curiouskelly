#!/usr/bin/env python3
"""
Interactive Audio Demonstration Script for Hybrid TTS System
Showcases all generated audio samples and system capabilities
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class AudioDemo:
    """Interactive audio demonstration class"""
    
    def __init__(self, test_audio_dir="test_audio_output", real_audio_dir="real_audio_output"):
        self.test_audio_dir = Path(test_audio_dir)
        self.real_audio_dir = Path(real_audio_dir)
        self.audio_files = {}
        self.load_audio_files()
    
    def load_audio_files(self):
        """Load all available audio files"""
        print("🔍 Loading audio files...")
        
        # Load test audio files
        if self.test_audio_dir.exists():
            for category_dir in self.test_audio_dir.iterdir():
                if category_dir.is_dir():
                    category_name = category_dir.name
                    self.audio_files[category_name] = []
                    
                    for audio_file in category_dir.glob("*.wav"):
                        metadata_file = audio_file.with_suffix('.json')
                        metadata = {}
                        if metadata_file.exists():
                            try:
                                with open(metadata_file, 'r', encoding='utf-8') as f:
                                    metadata = json.load(f)
                            except:
                                pass
                        
                        self.audio_files[category_name].append({
                            'file': audio_file,
                            'metadata': metadata,
                            'name': audio_file.name
                        })
        
        # Load real audio files
        if self.real_audio_dir.exists():
            for category_dir in self.real_audio_dir.iterdir():
                if category_dir.is_dir():
                    category_name = f"real_{category_dir.name}"
                    self.audio_files[category_name] = []
                    
                    for audio_file in category_dir.glob("*.wav"):
                        metadata_file = audio_file.with_suffix('.json')
                        metadata = {}
                        if metadata_file.exists():
                            try:
                                with open(metadata_file, 'r', encoding='utf-8') as f:
                                    metadata = json.load(f)
                            except:
                                pass
                        
                        self.audio_files[category_name].append({
                            'file': audio_file,
                            'metadata': metadata,
                            'name': audio_file.name
                        })
        
        print(f"✅ Loaded {sum(len(files) for files in self.audio_files.values())} audio files")
    
    def show_main_menu(self):
        """Display main menu"""
        print("\n" + "="*60)
        print("🎵 HYBRID TTS SYSTEM - AUDIO DEMONSTRATION")
        print("="*60)
        print("\nAvailable Categories:")
        
        categories = list(self.audio_files.keys())
        for i, category in enumerate(categories, 1):
            file_count = len(self.audio_files[category])
            print(f"  {i:2d}. {category.replace('_', ' ').title()} ({file_count} files)")
        
        print(f"\n  {len(categories)+1:2d}. Random Sample")
        print(f"  {len(categories)+2:2d}. System Overview")
        print(f"  {len(categories)+3:2d}. Voice Comparison")
        print(f"  {len(categories)+4:2d}. Quality Analysis")
        print(f"  {len(categories)+5:2d}. Exit")
        
        return categories
    
    def show_category_menu(self, category):
        """Display category-specific menu"""
        files = self.audio_files[category]
        print(f"\n📁 {category.replace('_', ' ').title()}")
        print("-" * 40)
        
        if not files:
            print("No audio files found in this category.")
            return
        
        # Group files by type for better organization
        file_groups = {}
        for file_info in files:
            name = file_info['name']
            if 'interpolation' in name:
                group = 'Voice Interpolation'
            elif 'morphing' in name:
                group = 'Voice Morphing'
            elif 'continuum' in name:
                group = 'Voice Continuum'
            elif 'family' in name:
                group = 'Voice Family'
            elif 'emotion' in name:
                group = 'Emotional Speech'
            elif 'prosody' in name:
                group = 'Prosody Control'
            elif 'quality' in name:
                group = 'Quality Tests'
            elif 'analysis' in name:
                group = 'Voice Analysis'
            else:
                group = 'Basic Synthesis'
            
            if group not in file_groups:
                file_groups[group] = []
            file_groups[group].append(file_info)
        
        # Display grouped files
        file_index = 1
        file_mapping = {}
        
        for group, group_files in file_groups.items():
            print(f"\n{group}:")
            for file_info in group_files:
                print(f"  {file_index:2d}. {file_info['name']}")
                file_mapping[file_index] = file_info
                file_index += 1
        
        print(f"\n  {file_index:2d}. Play All in Category")
        print(f"  {file_index+1:2d}. Back to Main Menu")
        
        return file_mapping, file_index
    
    def play_audio_file(self, file_info):
        """Play an audio file (simulated)"""
        print(f"\n🎵 Playing: {file_info['name']}")
        print(f"📁 File: {file_info['file']}")
        
        # Display metadata if available
        if file_info['metadata']:
            print("\n📊 Metadata:")
            for key, value in file_info['metadata'].items():
                if key != 'text':  # Skip text to keep output clean
                    print(f"  {key}: {value}")
        
        # Simulate audio playback
        print("\n🔊 [Audio playback simulation - 3 seconds]")
        time.sleep(1)
        print("   ████████████████████████████████████████")
        time.sleep(1)
        print("   ████████████████████████████████████████")
        time.sleep(1)
        print("   ████████████████████████████████████████")
        print("✅ Playback complete!")
    
    def play_random_sample(self):
        """Play a random audio sample"""
        all_files = []
        for category_files in self.audio_files.values():
            all_files.extend(category_files)
        
        if not all_files:
            print("No audio files available.")
            return
        
        random_file = random.choice(all_files)
        print(f"\n🎲 Random Sample from: {random_file['file'].parent.name}")
        self.play_audio_file(random_file)
    
    def show_system_overview(self):
        """Display system overview and statistics"""
        print("\n📊 SYSTEM OVERVIEW")
        print("="*40)
        
        total_files = sum(len(files) for files in self.audio_files.values())
        print(f"Total Audio Files: {total_files}")
        
        print("\nCategories:")
        for category, files in self.audio_files.items():
            print(f"  {category.replace('_', ' ').title()}: {len(files)} files")
        
        print("\n🎯 Key Features Demonstrated:")
        print("  • Voice Interpolation - Smooth transitions between voices")
        print("  • Voice Morphing - Advanced voice transformation")
        print("  • Voice Continuum - Navigation through voice space")
        print("  • Voice Family - Related voice variations")
        print("  • Emotional Speech - Context-aware emotional expression")
        print("  • Prosody Control - Fine-grained speech parameters")
        print("  • Quality Assessment - Multiple quality levels")
        print("  • Voice Analysis - Comprehensive voice characteristics")
        
        print("\n🔧 Technical Capabilities:")
        print("  • Multi-Architecture Support (FastPitch, Tacotron2, HiFi-GAN, WaveGlow)")
        print("  • Real-time Voice Switching")
        print("  • Hybrid Data Generation (Real + Synthetic)")
        print("  • Advanced Prosody Control")
        print("  • Voice Quality Metrics")
        print("  • Voice Similarity Measurement")
        print("  • Voice Clustering and Family Generation")
    
    def show_voice_comparison(self):
        """Show voice comparison between different voices"""
        print("\n🔄 VOICE COMPARISON")
        print("="*40)
        
        # Find interpolation files for comparison
        interpolation_files = []
        for category, files in self.audio_files.items():
            if 'interpolation' in category:
                interpolation_files.extend(files)
        
        if not interpolation_files:
            print("No interpolation files found for voice comparison.")
            return
        
        # Group by voice pairs
        voice_pairs = {}
        for file_info in interpolation_files:
            name = file_info['name']
            if 'voice_1_voice_2' in name:
                pair = 'Voice 1 ↔ Voice 2'
            elif 'voice_2_voice_3' in name:
                pair = 'Voice 2 ↔ Voice 3'
            elif 'voice_1_voice_3' in name:
                pair = 'Voice 1 ↔ Voice 3'
            else:
                continue
            
            if pair not in voice_pairs:
                voice_pairs[pair] = []
            voice_pairs[pair].append(file_info)
        
        print("Available Voice Comparisons:")
        for i, (pair, files) in enumerate(voice_pairs.items(), 1):
            print(f"  {i}. {pair} ({len(files)} interpolation steps)")
        
        print(f"\n  {len(voice_pairs)+1}. Back to Main Menu")
        
        try:
            choice = int(input("\nSelect voice pair to compare: ")) - 1
            if 0 <= choice < len(voice_pairs):
                pair_name = list(voice_pairs.keys())[choice]
                files = voice_pairs[pair_name]
                
                print(f"\n🎵 Comparing {pair_name}")
                print("Playing interpolation sequence...")
                
                # Sort files by interpolation weight
                sorted_files = sorted(files, key=lambda x: x['name'])
                
                for file_info in sorted_files:
                    self.play_audio_file(file_info)
                    time.sleep(0.5)
            else:
                print("Invalid selection.")
        except (ValueError, IndexError):
            print("Invalid input.")
    
    def show_quality_analysis(self):
        """Show quality analysis of different audio samples"""
        print("\n🔍 QUALITY ANALYSIS")
        print("="*40)
        
        # Find quality test files
        quality_files = []
        for category, files in self.audio_files.items():
            if 'quality' in category:
                quality_files.extend(files)
        
        if not quality_files:
            print("No quality test files found.")
            return
        
        print("Quality Test Samples:")
        for i, file_info in enumerate(quality_files, 1):
            name = file_info['name']
            quality_level = name.split('_')[1].replace('_', ' ').title()
            print(f"  {i}. {quality_level}")
        
        print(f"\n  {len(quality_files)+1}. Play All Quality Levels")
        print(f"  {len(quality_files)+2}. Back to Main Menu")
        
        try:
            choice = int(input("\nSelect quality level: ")) - 1
            if 0 <= choice < len(quality_files):
                self.play_audio_file(quality_files[choice])
            elif choice == len(quality_files):
                print("\n🎵 Playing all quality levels for comparison...")
                for file_info in quality_files:
                    self.play_audio_file(file_info)
                    time.sleep(0.5)
            else:
                print("Invalid selection.")
        except (ValueError, IndexError):
            print("Invalid input.")
    
    def run(self):
        """Run the interactive demonstration"""
        print("🎵 Welcome to the Hybrid TTS System Audio Demonstration!")
        
        while True:
            categories = self.show_main_menu()
            
            try:
                choice = int(input("\nSelect an option: ")) - 1
                
                if choice < len(categories):
                    # Category selected
                    category = categories[choice]
                    file_mapping, back_index = self.show_category_menu(category)
                    
                    while True:
                        try:
                            sub_choice = int(input(f"\nSelect file (1-{back_index-1}) or {back_index} for all, {back_index+1} for back: ")) - 1
                            
                            if sub_choice < len(file_mapping):
                                # Individual file selected
                                file_info = file_mapping[sub_choice + 1]
                                self.play_audio_file(file_info)
                            elif sub_choice == len(file_mapping):
                                # Play all files in category
                                print(f"\n🎵 Playing all files in {category}...")
                                for file_info in file_mapping.values():
                                    self.play_audio_file(file_info)
                                    time.sleep(0.5)
                                break
                            elif sub_choice == len(file_mapping) + 1:
                                # Back to main menu
                                break
                            else:
                                print("Invalid selection.")
                        except (ValueError, IndexError):
                            print("Invalid input.")
                
                elif choice == len(categories):
                    # Random sample
                    self.play_random_sample()
                elif choice == len(categories) + 1:
                    # System overview
                    self.show_system_overview()
                elif choice == len(categories) + 2:
                    # Voice comparison
                    self.show_voice_comparison()
                elif choice == len(categories) + 3:
                    # Quality analysis
                    self.show_quality_analysis()
                elif choice == len(categories) + 4:
                    # Exit
                    print("\n👋 Thank you for exploring the Hybrid TTS System!")
                    break
                else:
                    print("Invalid selection.")
                    
            except (ValueError, IndexError):
                print("Invalid input.")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Interactive audio demonstration for Hybrid TTS System")
    parser.add_argument("--test-audio-dir", default="test_audio_output", 
                       help="Directory containing test audio files")
    parser.add_argument("--real-audio-dir", default="real_audio_output", 
                       help="Directory containing real audio files")
    parser.add_argument("--no-interactive", action="store_true",
                       help="Run in non-interactive mode (show overview only)")
    
    args = parser.parse_args()
    
    demo = AudioDemo(args.test_audio_dir, args.real_audio_dir)
    
    if args.no_interactive:
        demo.show_system_overview()
    else:
        demo.run()

if __name__ == "__main__":
    main()
