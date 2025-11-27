"""
iClone Animation Batch Export via Python API
Compatible with iClone 8.4+
"""

import RLPy
import os
import time
from datetime import datetime

# Configuration - EDIT THIS PATH TO MATCH YOUR PROJECT FOLDER
PROJECT_ROOT = "C:/Kelly_Projects/Lessons" 
OUTPUT_DIR = "C:/Kelly_Animations/Exports"

# If you don't have projects yet, use the current open project
EXPORT_CURRENT_ONLY = True

def export_current_animation(output_path):
    """Export the currently open project animation."""
    
    print(f"Exporting current animation to: {output_path}")
    
    # Get current time range
    timeline = RLPy.RGlobal.GetTime()
    start_time = RLPy.RTime(0)
    end_time = timeline.GetEndTime()
    
    # FBX Export Options
    export_setting = RLPy.EExportFbxOptions()
    
    # Essential settings for Unity
    export_setting.SetOption(RLPy.EFbxExportOption_Mesh, False)  # Animation only (smaller files)
    export_setting.SetOption(RLPy.EFbxExportOption_Skin, False)
    export_setting.SetOption(RLPy.EFbxExportOption_BlendShape, True) # Critical for face
    export_setting.SetOption(RLPy.EFbxExportOption_Animation, True)
    export_setting.SetOption(RLPy.EFbxExportOption_Material, False)
    export_setting.SetOption(RLPy.EFbxExportOption_Texture, False)
    
    # Coordinate System (Unity = Y Up)
    export_setting.SetCoordinate(RLPy.ECoordinateSystem_Unity)
    export_setting.SetTimeRange(start_time, end_time)
    
    # FPS (Unity WebGL standard is usually 30 or 60)
    export_setting.SetFPS(RLPy.EExportFPS_60)
    
    # Export
    try:
        result = RLPy.RFileIO.ExportFbxFile(output_path, export_setting)
        if result == RLPy.RStatus.Success:
            print("✓ Export Successful")
            return True
        else:
            print("✗ Export Failed (API Error)")
            return False
    except Exception as e:
        print(f"✗ Export Failed: {str(e)}")
        return False

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if EXPORT_CURRENT_ONLY:
        # Just export what is currently open in iClone
        filename = "Kelly_Current_Animation.fbx"
        full_path = os.path.join(OUTPUT_DIR, filename)
        export_current_animation(full_path)
    else:
        # Batch mode (requires valid paths)
        print("Batch mode not configured yet. Set EXPORT_CURRENT_ONLY = True")

if __name__ == "__main__":
    main()
