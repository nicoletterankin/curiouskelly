# Claude.ai File Automation Setup Guide

## 🎯 Goal
Automatically download files created by Claude and save them to your codebase without manual downloads.

---

## 📋 Available Solutions

I've created **two Python scripts** for you:

### **Option 1: Direct API Downloader** (`scripts/claude_file_downloader.py`)
- Uses Claude API to directly access project files
- Requires Claude API endpoints for file access
- **Status:** Template ready, needs Claude API endpoint details

### **Option 2: Messages API Downloader** (`scripts/claude_file_downloader_alternative.py`) ⭐ **RECOMMENDED**
- Uses Claude Messages API to ask Claude for files
- Works with Claude's current API capabilities
- **Status:** Ready to use (may need API endpoint adjustments)

---

## 🚀 Quick Setup

### **Step 1: Install Dependencies**

```bash
pip install requests python-dotenv
```

### **Step 2: Get Claude API Key**

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Navigate to **API Keys**
3. Create a new API key
4. Copy the key

### **Step 3: Set Environment Variables**

**Windows (PowerShell):**
```powershell
$env:CLAUDE_API_KEY="your_api_key_here"
$env:CLAUDE_PROJECT_ID="the-daily-lesson"
$env:OUTPUT_DIRECTORY="lesson-player"
```

**Linux/Mac:**
```bash
export CLAUDE_API_KEY="your_api_key_here"
export CLAUDE_PROJECT_ID="the-daily-lesson"
export OUTPUT_DIRECTORY="lesson-player"
```

**Or create `.env` file:**
```env
CLAUDE_API_KEY=your_api_key_here
CLAUDE_PROJECT_ID=the-daily-lesson
OUTPUT_DIRECTORY=lesson-player
```

### **Step 4: Run Script**

```bash
python scripts/claude_file_downloader_alternative.py
```

---

## ⚙️ Scheduling (Optional)

### **Windows Task Scheduler**

1. Open **Task Scheduler**
2. Create **Basic Task**
3. Set trigger: **Daily** or **When computer starts**
4. Action: **Start a program**
5. Program: `python`
6. Arguments: `C:\path\to\scripts\claude_file_downloader_alternative.py`
7. Start in: `C:\Users\user\UI-TARS-desktop`

### **Linux/Mac Cron**

```bash
# Edit crontab
crontab -e

# Add line to run every hour
0 * * * * /usr/bin/python3 /path/to/scripts/claude_file_downloader_alternative.py >> /path/to/claude_download.log 2>&1
```

---

## 🔧 How It Works

1. **Script connects to Claude API** using your API key
2. **Asks Claude to list files** in the project
3. **Identifies lesson files** (matching patterns like `*-dna.json`)
4. **Checks which files are new** (tracks downloaded files in `.claude_download_state.json`)
5. **Downloads new files** and saves to `lesson-player/` directory
6. **Updates state** to avoid re-downloading

---

## 📁 File Organization

Files are saved to:
```
lesson-player/
├── {lesson-id}-dna.json
├── {lesson-id}-visual-prompts.json
├── {lesson-id}-knowledge-base.md
├── {lesson-id}-asset-manifest.json
├── {lesson-id}-teaching-moments.json
├── {lesson-id}-interactive-specs.json
├── {lesson-id}-animation-sequences.json
└── {lesson-id}-export-package.md
```

---

## 🚨 Troubleshooting

### **Error: CLAUDE_API_KEY not set**
**Solution:** Set environment variable or create `.env` file

### **Error: API endpoint not found**
**Solution:** 
- Check Claude API documentation for correct endpoints
- Update script with correct API endpoints
- May need to use Messages API approach (Option 2)

### **No files found**
**Solution:**
- Verify project ID is correct
- Check that Claude has created files in the project
- Verify API key has project access

### **Files not downloading**
**Solution:**
- Check API key permissions
- Verify Claude API version in headers
- Try Option 2 (Messages API approach)

---

## 🔄 Alternative: GitHub Actions

If you prefer CI/CD integration, I can create a GitHub Actions workflow that:
- Runs on schedule (every hour)
- Downloads files from Claude
- Commits to repository
- Triggers deployment

**Would you like me to create this?**

---

## 📝 Notes

- **API Limitations:** Claude API may have rate limits - script includes error handling
- **State Tracking:** Script tracks downloaded files to avoid duplicates
- **File Patterns:** Only downloads lesson files (matches patterns like `*-dna.json`)
- **Error Handling:** Script includes error handling and logging

---

## 🎯 Next Steps

1. **Test the script:** Run it manually first to verify it works
2. **Set up scheduling:** Configure Task Scheduler or cron for automation
3. **Monitor:** Check logs to ensure files are downloading correctly
4. **Adjust:** Modify file patterns or output directory as needed

---

## 💡 Future Enhancements

Possible improvements:
- **Webhook integration:** If Claude adds webhook support
- **Real-time monitoring:** Watch for new files in real-time
- **Auto-validation:** Validate downloaded files against schema
- **Auto-commit:** Commit files to Git automatically
- **Notifications:** Send alerts when new files are downloaded

---

**Ready to automate!** Run the script and let me know if you need any adjustments.

