# How to Find Your Claude API Key

## 🎯 Quick Answer

Your Claude API key is found in the **Anthropic Console**, not in the Claude desktop app or GitHub integration.

---

## 📋 Step-by-Step Instructions

### **Method 1: Anthropic Console** (Recommended)

1. **Go to Anthropic Console**
   - Visit: [console.anthropic.com](https://console.anthropic.com)
   - Sign in with your Anthropic account
   - (If you don't have an account, sign up first)

2. **Navigate to API Keys**
   - Click **"API Keys"** in the left sidebar
   - Or go directly to: [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)

3. **Create or View API Key**
   - If you see existing keys, they'll be listed (partially masked)
   - To create a new key:
     - Click **"Create Key"** button
     - Give it a name (e.g., "Lesson Automation")
     - Click **"Create Key"**
     - **IMPORTANT:** Copy the key immediately - you won't see it again!
     - Format: `sk-ant-api03-...` (starts with `sk-ant-api03-`)

4. **Copy Your Key**
   - The key will look like: `sk-ant-api03-ABC123...XYZ789`
   - Copy the entire key
   - Store it securely (password manager recommended)

---

## 🔍 Where It's NOT Located

### ❌ **NOT in Claude Desktop App**
- The files in `C:\Users\user\AppData\Roaming\Claude` are local app data
- These don't contain your API key
- The desktop app uses its own authentication

### ❌ **NOT in GitHub Claude Integration**
- The GitHub app integration (shown in your screenshot) is for GitHub-specific features
- This doesn't give you an API key
- It's a separate integration

### ❌ **NOT in Claude Code CLI**
- The `claude.exe` CLI tool you installed is for GitHub integration
- It doesn't provide API keys
- It's for running Claude Code in GitHub workflows

---

## ✅ Where It IS Located

### **Anthropic Console** → **API Keys**
- **URL:** [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
- **What you'll see:**
  - List of API keys (if any exist)
  - "Create Key" button
  - Key names and creation dates
  - (Keys are partially masked for security)

---

## 🚨 If You Don't Have an Account

1. **Sign Up for Anthropic**
   - Go to [console.anthropic.com](https://console.anthropic.com)
   - Click **"Sign Up"** or **"Get Started"**
   - Create an account with your email
   - Verify your email address

2. **Set Up Billing** (if required)
   - Some API features may require billing setup
   - Follow the prompts in the console

3. **Create API Key**
   - Once logged in, go to **API Keys** section
   - Create your first key

---

## 🔑 API Key Format

Your API key will look like:
```
sk-ant-api03-ABC123DEF456GHI789JKL012MNO345PQR678STU901VWX234YZ567
```

**Characteristics:**
- Starts with `sk-ant-api03-`
- Long string of characters (usually 100+ characters)
- Shown only once when created
- Can be regenerated if lost

---

## 💾 How to Store Your API Key

### **Option 1: Environment Variable** (Recommended for scripts)

**Windows PowerShell:**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"
```

**Windows Command Prompt:**
```cmd
set ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

**Linux/Mac:**
```bash
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"
```

### **Option 2: .env File** (Recommended for projects)

Create `.env` file in your project root:
```env
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
CLAUDE_PROJECT_ID=the-daily-lesson
OUTPUT_DIRECTORY=lesson-player
```

**⚠️ Important:** Add `.env` to `.gitignore` to avoid committing your key!

### **Option 3: Password Manager**
- Store in 1Password, LastPass, Bitwarden, etc.
- Copy when needed
- More secure than plain text files

---

## 🧪 Test Your API Key

Once you have your key, test it:

**Python:**
```python
import os
from anthropic import Anthropic

api_key = os.getenv("ANTHROPIC_API_KEY")
if api_key:
    client = Anthropic(api_key=api_key)
    print("✅ API key is set!")
    # Test API call
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=10,
        messages=[{"role": "user", "content": "Hi"}]
    )
    print("✅ API is working!")
else:
    print("❌ API key not found")
```

**Or use our script:**
```bash
python scripts/claude_api_downloader.py
```

---

## 🆘 Troubleshooting

### **"I can't find API Keys section"**
- Make sure you're logged into [console.anthropic.com](https://console.anthropic.com)
- Look for "Settings" or "API" in the navigation
- Try direct link: [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)

### **"I don't have an Anthropic account"**
- Sign up at [console.anthropic.com](https://console.anthropic.com)
- You'll need to create an account to get API access

### **"I lost my API key"**
- Go to API Keys section
- Delete the old key (if visible)
- Create a new key
- Update your scripts/environment with the new key

### **"API key doesn't work"**
- Verify you copied the entire key (no spaces, no line breaks)
- Check it starts with `sk-ant-api03-`
- Make sure you're using the correct environment variable name
- Try regenerating the key

---

## 📚 Alternative: Use Export Processor Instead

If you can't get an API key right now, you can use the **export processor** approach:

1. **Export conversation** from Claude.ai (web interface)
2. **Run export processor:**
   ```bash
   python scripts/claude_export_helper.py exported_conversation.txt
   ```
3. **Files automatically extracted** → Saved to `lesson-player/`

This doesn't require an API key!

---

## 🎯 Quick Checklist

- [ ] Go to [console.anthropic.com](https://console.anthropic.com)
- [ ] Sign in (or create account)
- [ ] Navigate to **API Keys** section
- [ ] Create new key (or copy existing)
- [ ] Store key securely
- [ ] Set environment variable or add to `.env`
- [ ] Test with script

---

## 🔗 Direct Links

- **Anthropic Console:** [console.anthropic.com](https://console.anthropic.com)
- **API Keys:** [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
- **API Documentation:** [docs.anthropic.com](https://docs.anthropic.com)

---

**Need more help?** Let me know what step you're stuck on!

