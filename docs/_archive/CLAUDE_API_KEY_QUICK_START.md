# Claude API Key - Quick Start Guide

## 🎯 Where to Get Your API Key

**Direct Link:** [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)

---

## ⚡ 3-Step Process

### **Step 1: Go to Anthropic Console**
1. Visit: [console.anthropic.com](https://console.anthropic.com)
2. Sign in (or create account if needed)

### **Step 2: Get API Key**
1. Click **"API Keys"** in left sidebar
2. Click **"Create Key"** button
3. Name it: "Lesson Automation"
4. **Copy the key immediately** (you won't see it again!)

### **Step 3: Use Your Key**
```bash
# Set environment variable
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"

# Or add to .env file
echo "ANTHROPIC_API_KEY=sk-ant-api03-your-key-here" >> .env
```

---

## 🔑 Key Format

Your key will look like:
```
sk-ant-api03-ABC123DEF456GHI789JKL012MNO345PQR678STU901VWX234YZ567
```

**Important:** 
- Starts with `sk-ant-api03-`
- Very long string (100+ characters)
- Copy the ENTIRE key

---

## ❌ Common Mistakes

- ❌ Looking in Claude desktop app (wrong place)
- ❌ Looking in GitHub integration (wrong place)
- ❌ Not copying the entire key
- ❌ Adding spaces or line breaks

---

## ✅ Quick Test

Once you have your key:
```bash
python scripts/claude_api_downloader.py
```

If it works, you'll see:
```
✅ API key is set!
✅ API is working!
```

---

## 🆘 Still Can't Find It?

**Option 1:** Use export processor instead (no API key needed)
```bash
python scripts/claude_export_helper.py exported_conversation.txt
```

**Option 2:** Check detailed guide: `HOW_TO_FIND_CLAUDE_API_KEY.md`

---

**Direct Link:** [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)

