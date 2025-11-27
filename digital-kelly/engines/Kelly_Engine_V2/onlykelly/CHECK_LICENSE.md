# How to Check Your CC/iC Unity Tools License Status

## Quick Check

### In Character Creator 5:
1. Open Character Creator 5
2. Go to: **Plugins → CC/iC Unity Tools → License Manager**
3. Look for license status:
   - ✅ "License: Active" = You have it!
   - ❌ "Trial Mode" or no option = Need to purchase

---

## Check Reallusion Member Area

### Step 1: Log In
1. Go to: https://www.reallusion.com/member/
2. Log in with your Reallusion account

### Step 2: Find Software Registration
1. Look for "Software Registration" or "My Products" section
2. Click to view your registered products

### Step 3: Look for Unity Tools
Search for any of these product names:
- "CC/iC Auto Setup Unity"
- "CC/iC Unity Pipeline Tools"
- "CC Unity Tools"
- "iClone Unity Tools"

### Step 4: Interpret Results

**If you see it listed with a serial number:**
```
✅ YOU HAVE THE LICENSE

Product: CC/iC Auto Setup Unity
Serial: XXXX-XXXX-XXXX-XXXX
Status: Registered

Next Steps:
1. Copy the serial number
2. Open CC5 → Plugins → License Manager
3. Enter serial number
4. Click Activate
5. Restart CC5
6. Export Kelly without watermark!
```

**If you DON'T see it listed:**
```
❌ YOU DON'T HAVE THE LICENSE

Your registered products show CC5, iClone 8, etc.
But NO CC/iC Unity Tools entry.

Options:
A) Purchase license (~$199)
   https://www.reallusion.com/auto-setup/unity/default.html

B) Launch with watermark, buy later
   Watermark is small, doesn't affect functionality
   Can remove post-launch when revenue flows
```

---

## Your Current License Status

Based on the project analysis:

| Product | Status |
|---------|--------|
| Character Creator 5 | ✅ Owned & Activated |
| iClone 8 | ✅ Owned & Activated |
| Headshot Plugin | ✅ Owned |
| AccuLips | ✅ Owned |
| CC/iC Unity Tools | ❌ **NOT OWNED** |

### The Watermark Source:
The "Trial Version" watermark in Kelly's WebGL build comes from using the **trial version** of CC/iC Unity Tools. The artist who originally set up the project didn't have the license.

---

## Purchase Options

### Option 1: Buy Now ($199)
**Best for:** Perfect launch, professional appearance

1. Go to: https://www.reallusion.com/auto-setup/unity/default.html
2. Click "Buy Now"
3. Complete purchase
4. Receive license key via email (instant)
5. Activate in CC5
6. Re-export Kelly
7. No more watermark!

### Option 2: Buy Later
**Best for:** Budget constraints, testing first

1. Launch with watermark (December 17)
2. Validate product-market fit
3. Generate initial revenue
4. Purchase license with revenue
5. Update Kelly post-launch
6. Remove watermark in v1.1

---

## Activation Steps (Once You Have License)

### Step 1: Get Your License Key
- Check email from Reallusion after purchase
- Or log into Member Area → Software Registration
- Copy the serial number

### Step 2: Activate in Character Creator 5
1. Open Character Creator 5
2. Menu: Plugins → CC/iC Unity Tools → License Manager
3. Paste your license key
4. Click **Activate**
5. Should show: "License activated successfully"

### Step 3: Activate in iClone 8 (Same Process)
1. Open iClone 8
2. Menu: Plugins → CC/iC Unity Tools → License Manager
3. Paste same license key
4. Click **Activate**

### Step 4: Restart Both Applications
- Close CC5 and iClone 8
- Reopen them
- License is now active

### Step 5: Re-Export Kelly
1. Open Kelly in iClone 8
2. Plugins → CC/iC Unity Tools → Send to Unity
3. Kelly exports WITHOUT watermark!

### Step 6: Rebuild in Unity
1. Delete old Kelly from scene
2. Add new Kelly export
3. Kelly → Build → Build WebGL (Production)
4. Deploy to Netlify
5. Verify: No watermark! 🎉

---

## FAQ

### Q: Can I use CC5/iClone licenses for Unity exports?
**A:** No. CC5 and iClone licenses are separate from the Unity Pipeline Tools license. You need to purchase CC/iC Unity Tools separately.

### Q: Is the watermark visible to end users?
**A:** Yes. The "Trial Version" text appears in the corner of the WebGL build. It's small but visible.

### Q: Does the watermark affect functionality?
**A:** No. Kelly still works fully - animations, blendshapes, materials all function. It's just a visual watermark.

### Q: Can I remove the watermark without buying?
**A:** No legitimate way. The watermark is embedded by the trial version of the Unity tools.

### Q: Is $199 a one-time or subscription?
**A:** One-time purchase. You own the license forever with free updates.

### Q: Can I use one license for multiple projects?
**A:** Yes. The license works across all your Unity projects.

---

## Decision Matrix

| Factor | Buy Now | Buy Later |
|--------|---------|-----------|
| Launch appearance | Professional | Has watermark |
| Upfront cost | $199 | $0 |
| Timeline risk | Low | None |
| Post-launch work | None | Update needed |
| Revenue impact | Possibly higher | Possibly lower |

### Recommendation:
**Hybrid Approach:**
1. Deploy TODAY with watermark (safety net)
2. Purchase license THIS WEEK
3. Re-export and redeploy BEFORE December 17
4. Launch with clean, professional version

---

*Last Updated: November 26, 2025*

