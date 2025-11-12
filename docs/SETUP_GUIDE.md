# CC5/iClone Pipeline Setup Guide

## Overview

This guide will help you set up a complete creative pipeline for Character Creator 5 (CC5) and iClone 8, enabling automated generation of photoreal talking avatars with voice synchronization.

## Prerequisites

- **Windows 10/11** with RTX GPU (RTX 5090 recommended)
- **Administrator privileges** for software installation
- **Internet connection** for downloading dependencies
- **Reallusion account** for software licensing

## Quick Setup (Automated)

### Option 1: Full Pipeline Execution
```powershell
# Run the complete pipeline (recommended)
cd C:\Users\user\Creative-Pipeline
.\scripts\run_pipeline.ps1
```

### Option 2: Step-by-Step Execution
```powershell
# 1. Bootstrap workspace
.\scripts\00_bootstrap.ps1

# 2. Install dependencies
.\scripts\01_install_deps.ps1

# 3. Detect Reallusion software (after installation)
.\scripts\02_detect_reallusion.ps1

# 4. Analyze audio (after placing audio file)
.\scripts\10_audio_analyze.ps1

# 5. Generate analytics (after rendering)
.\scripts\20_contact_sheet.ps1
.\scripts\21_frame_metrics.ps1

# 6. Scaffold character projects
.\scripts\30_new_character.ps1

# 7. Create VS Code tasks
.\scripts\40_write_tasksjson.ps1
```

## Manual Setup Steps

### 1. Software Installation

**Required Software:**
- Character Creator 5 (CC5)
- Headshot 2 (CC5 compatible)
- iClone 8
- Motion LIVE hub
- (Optional) AccuFACE for facial capture
- (Optional) Live Face for iPhone TrueDepth

**Installation Process:**
1. Download Reallusion Hub from [reallusion.com](https://www.reallusion.com)
2. Install and sign in to Reallusion Hub
3. Purchase and install the required software through the Hub
4. Activate all licenses

### 2. Workspace Configuration

**Directory Structure:**
The pipeline creates the following structure on `D:\iLearnStudio`:
```
D:\iLearnStudio\
├── config\                    # Configuration files
├── installers\               # Software installers
├── scripts\                  # PowerShell automation
├── tools\                    # Python analysis tools
├── docs\                     # Documentation
├── metrics\                  # Status tracking
├── analytics\Kelly\          # Kelly analytics
├── projects\_Shared\         # Shared assets
├── projects\Kelly\           # Kelly character
└── renders\Kelly\            # Rendered outputs
```

### 3. Content Setup

**Audio File:**
- Place your audio file as `projects\Kelly\Audio\kelly25_audio.wav`
- Supported formats: WAV, MP3, M4A
- Recommended: 44.1kHz, 16-bit, mono or stereo

**Reference Video (Optional):**
- Place reference video as `projects\Kelly\Ref\kelly_ref_video.mp4`
- Used for AccuFACE facial capture
- Should show clear facial expressions

### 4. Character Creation Workflow

**In Character Creator 5:**
1. Create new HD Character
2. Use Headshot 2 with front-facing photo
3. Adjust facial features (jaw, lips, nose, eyes)
4. Save as `projects\Kelly\CC5\Kelly_HD_Head.ccProject`

**In iClone 8:**
1. Open `projects\_Shared\iClone\DirectorsChair_Template.iProject`
2. Import Kelly character from CC5
3. Set up AccuLips with audio file
4. (Optional) Apply AccuFACE with reference video
5. Render test video to `renders\Kelly\kelly_test_talk_v1.mp4`

## VS Code/Cursor Integration

After running the setup scripts, you can use VS Code or Cursor tasks:

1. Open Command Palette (`Ctrl+Shift+P`)
2. Type "Tasks: Run Task"
3. Select from available automation tasks:
   - Bootstrap Repo
   - Install Deps
   - Detect Reallusion Installs
   - Analyze Audio (Kelly)
   - Contact Sheet (Kelly)
   - Frame Metrics (Kelly)
   - Scaffold All Characters

## Troubleshooting

### Common Issues

**PowerShell Execution Policy:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Missing Dependencies:**
- Ensure Python 3.11 is installed
- Verify FFmpeg is in PATH
- Check Git LFS is properly configured

**Reallusion Software Not Detected:**
- Verify software is installed in default locations
- Check license activation
- Run as administrator if needed

**Audio Analysis Fails:**
- Verify audio file exists and is accessible
- Check audio format compatibility
- Ensure Python libraries are installed

### Validation

Run the validation script to check your setup:
```powershell
.\scripts\validate_setup.ps1
```

## Success Criteria

Your setup is complete when:
- [ ] All directories created successfully
- [ ] Dependencies installed and working
- [ ] Reallusion software detected and licensed
- [ ] Audio analysis generates plots and metrics
- [ ] Kelly character renders successfully
- [ ] Analytics scripts produce expected outputs
- [ ] VS Code tasks are available and functional

## Next Steps

1. **Create your first character** following the Kelly workflow
2. **Customize the pipeline** for your specific needs
3. **Scale to multiple characters** using the scaffolding system
4. **Integrate with your existing workflow** using the automation scripts

## Support

For issues or questions:
- Check the troubleshooting section above
- Review the automation logs in `metrics\`
- Consult the Reallusion documentation
- Check the project README for updates

---

*This pipeline is designed for AI agents and automation to execute creative workflows autonomously.*
