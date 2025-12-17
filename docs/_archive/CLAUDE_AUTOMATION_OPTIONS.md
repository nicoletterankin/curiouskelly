# Claude.ai File Automation Options

## 🎯 Goal
Automatically download files created by Claude and save them to your codebase without manual downloads.

---

## 🔍 Available Automation Methods

### **Option 1: Claude API + Python Script** ⭐ RECOMMENDED

**How it works:**
- Use Claude's API to programmatically access project files
- Python script runs periodically (cron/scheduled task)
- Downloads new files and saves to `lesson-player/` directory

**Pros:**
- ✅ Fully automated
- ✅ Can run on schedule
- ✅ Direct API access
- ✅ Can filter for new files only

**Cons:**
- ⚠️ Requires Claude API key
- ⚠️ Need to set up scheduling

**Implementation:** See `scripts/claude_file_downloader.py` (will be created)

---

### **Option 2: GitHub Actions Workflow** ⭐ GOOD FOR CI/CD

**How it works:**
- GitHub Actions workflow runs on schedule
- Uses Claude API to check for new files
- Downloads and commits files to repository
- Auto-deploys if configured

**Pros:**
- ✅ Integrates with existing GitHub workflow
- ✅ Automatic commits
- ✅ Can trigger deployments
- ✅ Version control built-in

**Cons:**
- ⚠️ Requires Claude API key in GitHub secrets
- ⚠️ Runs on GitHub's schedule (not real-time)

**Implementation:** See `.github/workflows/claude-file-sync.yml` (will be created)

---

### **Option 3: Claude MCP Server** ⭐ ADVANCED

**How it works:**
- Set up Model Context Protocol (MCP) server
- Claude can directly write files to your codebase
- Real-time file saving during conversation

**Pros:**
- ✅ Real-time saving
- ✅ Claude writes directly to codebase
- ✅ No intermediate downloads needed

**Cons:**
- ⚠️ More complex setup
- ⚠️ Requires MCP server development
- ⚠️ Security considerations

**Implementation:** See `mcp-servers/claude-file-writer/` (will be created)

---

### **Option 4: Browser Extension + Local Script** ⭐ SIMPLE

**How it works:**
- Browser extension monitors Claude.ai project files tab
- Detects new files
- Triggers local script to download and organize

**Pros:**
- ✅ No API key needed
- ✅ Works with existing Claude.ai interface
- ✅ Simple setup

**Cons:**
- ⚠️ Requires browser extension
- ⚠️ Must have browser open
- ⚠️ Less reliable than API

**Implementation:** See `browser-extension/claude-file-monitor/` (will be created)

---

## 🚀 Recommended Solution: Option 1 (Python Script)

**Why:** Most reliable, fully automated, works offline, easy to schedule.

---

## 📋 Implementation: Python Script

### **Step 1: Install Dependencies**

```bash
pip install anthropic requests python-dotenv
```

### **Step 2: Set Up Environment Variables**

Create `.env` file:
```env
CLAUDE_API_KEY=your_api_key_here
CLAUDE_PROJECT_ID=your_project_id_here
OUTPUT_DIRECTORY=lesson-player
```

### **Step 3: Run Script**

```bash
python scripts/claude_file_downloader.py
```

### **Step 4: Schedule (Optional)**

**Windows (Task Scheduler):**
- Create scheduled task to run script every hour

**Linux/Mac (Cron):**
```bash
# Run every hour
0 * * * * /usr/bin/python3 /path/to/scripts/claude_file_downloader.py
```

---

## 📋 Implementation: GitHub Actions

### **Step 1: Add GitHub Secret**

1. Go to GitHub repository → Settings → Secrets and variables → Actions
2. Add secret: `CLAUDE_API_KEY`
3. Add secret: `CLAUDE_PROJECT_ID`

### **Step 2: Create Workflow File**

See `.github/workflows/claude-file-sync.yml` (will be created)

### **Step 3: Workflow Runs**

- Runs every hour automatically
- Downloads new files
- Commits to repository
- Triggers deployment if configured

---

## 🔧 Next Steps

I'll create the automation scripts for you. Which option would you prefer?

1. **Python Script** (Option 1) - Recommended for local automation
2. **GitHub Actions** (Option 2) - Recommended for CI/CD integration
3. **Both** - Maximum flexibility

---

## 📝 Notes

- **Claude API Access:** You'll need an Anthropic API key with project access
- **File Filtering:** Scripts will only download lesson files (matching `*-dna.json`, `*-visual-prompts.json`, etc.)
- **Organization:** Files will be saved to `lesson-player/` directory with proper naming
- **Error Handling:** Scripts include error handling and logging

---

**Ready to implement?** Let me know which option(s) you'd like, and I'll create the automation scripts!

