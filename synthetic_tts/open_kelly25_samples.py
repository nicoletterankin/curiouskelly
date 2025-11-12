#!/usr/bin/env python3
"""
Open Kelly25 Samples HTML Player
Create a desktop shortcut and open the HTML player automatically
"""

import os
import subprocess
import webbrowser
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_desktop_shortcut():
    """Create a desktop shortcut to the Kelly25 samples HTML file"""
    
    # Get current directory and HTML file path
    current_dir = Path.cwd()
    html_file = current_dir / "kelly25_voice_samples" / "kelly25_samples.html"
    
    # Get desktop path
    desktop_path = Path.home() / "Desktop"
    
    # Create shortcut file (Windows .lnk format)
    shortcut_path = desktop_path / "Kelly25 Voice Samples.lnk"
    
    try:
        # Create a simple batch file that opens the HTML
        batch_content = f'''@echo off
echo Opening Kelly25 Voice Samples...
start "" "{html_file}"
'''
        
        batch_file = desktop_path / "Kelly25_Voice_Samples.bat"
        with open(batch_file, 'w') as f:
            f.write(batch_content)
        
        logger.info(f"Desktop shortcut created: {batch_file}")
        return str(batch_file)
        
    except Exception as e:
        logger.error(f"Failed to create desktop shortcut: {e}")
        return None

def open_html_player():
    """Open the Kelly25 samples HTML player in the default browser"""
    
    # Get HTML file path
    html_file = Path.cwd() / "kelly25_voice_samples" / "kelly25_samples.html"
    
    if not html_file.exists():
        logger.error(f"HTML file not found: {html_file}")
        return False
    
    try:
        # Convert to file URL
        html_url = html_file.as_uri()
        
        # Open in default browser
        webbrowser.open(html_url)
        
        logger.info(f"Opened Kelly25 samples in browser: {html_url}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to open HTML file: {e}")
        return False

def main():
    """Main function to create shortcut and open HTML player"""
    
    print("🎵 Kelly25 Voice Samples - Opening HTML Player")
    print("=" * 60)
    
    # Get absolute path to HTML file
    html_file = Path.cwd() / "kelly25_voice_samples" / "kelly25_samples.html"
    
    print(f"HTML File: {html_file}")
    print(f"Exists: {html_file.exists()}")
    
    if html_file.exists():
        print(f"File Size: {html_file.stat().st_size:,} bytes")
        print(f"Last Modified: {html_file.stat().st_mtime}")
    
    # Create desktop shortcut
    print("\n📋 Creating Desktop Shortcut...")
    shortcut = create_desktop_shortcut()
    
    if shortcut:
        print(f"✅ Desktop shortcut created: {shortcut}")
    else:
        print("❌ Failed to create desktop shortcut")
    
    # Open HTML player
    print("\n🌐 Opening HTML Player in Browser...")
    success = open_html_player()
    
    if success:
        print("✅ Kelly25 samples opened in your default browser!")
        print("\n🎧 You can now:")
        print("   • Listen to all 40 voice samples")
        print("   • Browse by category (greetings, teaching, etc.)")
        print("   • Download WAV or MP3 files")
        print("   • Share the HTML file with others")
    else:
        print("❌ Failed to open HTML player")
        print(f"\n📁 Manual access: {html_file}")
        print("   Right-click the file and select 'Open with' your web browser")
    
    # Show file information
    print(f"\n📊 File Information:")
    print(f"   Location: {html_file}")
    print(f"   Size: {html_file.stat().st_size:,} bytes")
    print(f"   Samples: 40 WAV + 40 MP3 files")
    print(f"   Categories: 10 conversation types")
    print(f"   Duration: 5 seconds per sample")
    
    return success

if __name__ == "__main__":
    main()




































