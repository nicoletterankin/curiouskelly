# Claude API Complete Guide

## 🎯 Overview

This guide covers how to use the Anthropic Claude API to automate lesson file downloads from Claude.ai projects.

---

## 📋 Prerequisites

1. **Claude Console Account**
   - Go to [console.anthropic.com](https://console.anthropic.com)
   - Sign up or log in
   - Navigate to **API Keys** section

2. **API Key**
   - Create a new API key
   - Copy and store securely (you won't see it again)
   - Format: `sk-ant-api03-...`

3. **Python Environment**
   - Python 3.8+
   - `anthropic` package (official SDK)
   - Or `requests` for direct HTTP calls

---

## 🔑 Getting Your API Key

### Step 1: Access Claude Console

1. Visit [console.anthropic.com](https://console.anthropic.com)
2. Sign in with your Anthropic account
3. Navigate to **Settings** → **API Keys**

### Step 2: Create API Key

1. Click **"Create Key"**
2. Give it a name (e.g., "Lesson Automation")
3. Copy the key immediately (shown only once)
4. Store securely

### Step 3: Set Up Environment

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"
```

**Linux/Mac:**
```bash
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"
```

**Or create `.env` file:**
```env
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
CLAUDE_PROJECT_ID=the-daily-lesson
OUTPUT_DIRECTORY=lesson-player
```

---

## 📚 Claude API Endpoints

### **Base URL:**
```
https://api.anthropic.com/v1
```

### **Available Endpoints:**

1. **Messages API** - Send messages to Claude
   ```
   POST /v1/messages
   ```

2. **Files API** - Upload/manage files
   ```
   POST /v1/files
   GET /v1/files/{file_id}
   ```

3. **Projects API** - Manage projects (if available)
   ```
   GET /v1/projects/{project_id}/files
   ```

**Note:** Project file access may require using Messages API with project context.

---

## 🚀 Using Claude API

### **Method 1: Official Python SDK** (Recommended)

**Install:**
```bash
pip install anthropic
```

**Example:**
```python
from anthropic import Anthropic

client = Anthropic(api_key="your-api-key")

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4096,
    messages=[
        {
            "role": "user",
            "content": "List all lesson files in this project"
        }
    ]
)

print(message.content)
```

### **Method 2: Direct HTTP Requests**

**Example:**
```python
import requests

headers = {
    "x-api-key": "your-api-key",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

response = requests.post(
    "https://api.anthropic.com/v1/messages",
    headers=headers,
    json={
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "messages": [{
            "role": "user",
            "content": "List all lesson files in this project"
        }]
    }
)

print(response.json())
```

---

## 🔧 API Limitations & Considerations

### **Current Limitations:**

1. **No Direct Project File Access**
   - Claude API doesn't have direct endpoints to list/download project files
   - Must use Messages API to ask Claude for file contents

2. **Rate Limits**
   - Free tier: Limited requests
   - Paid tier: Higher limits
   - Check your plan limits

3. **Context Windows**
   - Claude 3.5 Sonnet: 200K tokens
   - Large files may need chunking

### **Workarounds:**

1. **Use Messages API** - Ask Claude to provide file contents
2. **Export + Process** - Export conversations, process with script
3. **Hybrid Approach** - Combine API calls with export processing

---

## 📝 Updated Automation Script

I'll create an updated script that uses the actual Claude API endpoints. See `scripts/claude_api_downloader.py` (will be created).

---

## 🎯 Recommended Workflow

### **Option A: Messages API Approach**

1. **Use Messages API** to ask Claude for file list
2. **Request file contents** one by one
3. **Save files** to codebase

**Pros:**
- ✅ Works with current API
- ✅ Fully automated
- ✅ No manual steps

**Cons:**
- ⚠️ Multiple API calls needed
- ⚠️ Rate limit considerations

### **Option B: Export + Process** (Most Reliable)

1. **Export conversation** from Claude.ai
2. **Run export processor** script
3. **Files extracted** automatically

**Pros:**
- ✅ Most reliable
- ✅ No API limitations
- ✅ Works immediately

**Cons:**
- ⚠️ Requires manual export step

---

## 🔐 Security Best Practices

1. **Never commit API keys** to Git
2. **Use environment variables** or `.env` files
3. **Rotate keys** periodically
4. **Use least privilege** - only grant necessary permissions
5. **Monitor usage** in Claude Console

---

## 📊 API Usage Monitoring

**Check Usage:**
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Navigate to **Usage** or **Billing**
3. View API call counts and costs

---

## 🚨 Troubleshooting

### **Error: Invalid API Key**
- Verify key starts with `sk-ant-api03-`
- Check for typos or extra spaces
- Regenerate key if needed

### **Error: Rate Limit Exceeded**
- Wait before retrying
- Upgrade plan if needed
- Implement exponential backoff

### **Error: Project Not Found**
- Verify project ID is correct
- Check project exists in Claude.ai
- May need to use Messages API instead

---

## 📚 Additional Resources

- **Official Docs:** [docs.anthropic.com](https://docs.anthropic.com)
- **API Reference:** [docs.anthropic.com/claude/reference](https://docs.anthropic.com/claude/reference)
- **Python SDK:** [github.com/anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python)

---

## 🎯 Next Steps

1. **Get API Key** - Follow steps above
2. **Test API** - Try a simple API call
3. **Set up automation** - Use updated scripts
4. **Monitor usage** - Check Claude Console

---

**Ready to automate!** Get your API key and we'll set up the automation scripts.

