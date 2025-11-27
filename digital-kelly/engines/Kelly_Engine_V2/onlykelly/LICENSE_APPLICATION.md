# CC/iC Unity Tools License Application Guide

## Overview

The "Trial Version" watermark visible in Kelly's WebGL build comes from using the trial version of **CC/iC Auto Setup Unity Tools** by Reallusion. This guide explains how to purchase and apply the license to remove the watermark.

---

## Purchase Information

**Product:** CC/iC Auto Setup for Unity
**Price:** ~$199 USD (one-time purchase)
**Purchase Link:** https://www.reallusion.com/auto-setup/unity/default.html

### What You Get:
- Removes "Trial Version" watermark from all builds
- Full commercial license for Unity projects
- Automatic material setup for CC/iC characters
- LOD generation tools
- Ongoing updates

---

## How to Apply License in Unity

### Step 1: Purchase License
1. Go to https://www.reallusion.com/auto-setup/unity/default.html
2. Click "Buy Now"
3. Complete purchase
4. Check email for license key (arrives within minutes)

### Step 2: Open Unity Project
1. Open Unity Hub
2. Open project: `digital-kelly/engines/Kelly_Engine_V2/onlykelly`
3. Wait for project to load completely

### Step 3: Enter License
1. In Unity top menu: **Reallusion > CC/iC Importer > License Manager**
   - (Or: **Window > Reallusion > License Manager**)
2. In the License Manager window:
   - Paste your license key from email
   - Click **Activate**
3. You should see: "License activated successfully"

### Step 4: Restart Unity
1. Close Unity completely
2. Reopen the project
3. The license is now active

### Step 5: Rebuild and Deploy
1. **Kelly > Build > 🚀 Build WebGL (Production)**
2. Wait for build to complete
3. Run: `.\deploy-kelly.ps1`
4. Verify watermark is gone at your Netlify URL

---

## Important Notes

- **No re-import needed:** You don't need to re-import Kelly after licensing
- **Just rebuild:** Simply rebuild WebGL and redeploy
- **One-time purchase:** License is perpetual, not subscription
- **Multiple projects:** License works across all your Unity projects

---

## Troubleshooting

### "License key invalid"
- Double-check you copied the entire key (no extra spaces)
- Ensure you're using the correct Reallusion account

### "License Manager not found"
- Ensure CC/iC Unity Tools package is installed
- Check: Window > Package Manager > Search "Reallusion"

### Watermark still appears after licensing
1. Close Unity completely
2. Delete `Library/` folder in project (forces reimport)
3. Reopen Unity
4. Rebuild WebGL

---

## Timeline Recommendation

**For December 17 Launch:**
- Launch WITH watermark (acceptable for soft launch)
- Purchase license after launch validation
- Remove watermark in post-launch update

**Cost-Benefit:**
- $199 is reasonable for production use
- Can defer until revenue is flowing
- Watermark doesn't affect functionality

---

## Contact

**Reallusion Support:** https://www.reallusion.com/support/
**License Issues:** support@reallusion.com

---

*Last Updated: November 26, 2025*

