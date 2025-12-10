# 🔑 Push Notification Credentials Checklist

**For: Nicolette Rankin (nicoletterankin@gmail.com)**  
**Date**: December 9, 2025  
**Launch**: December 17, 2025

---

## ✅ Already Complete

| Item | Status | Value |
|------|--------|-------|
| VAPID Public Key | ✅ | `BEgmu91QD3hye9UZ9MM6xZxfbIRrmhiKE3cV3XkfvxAlRMATdRY4skdaFAMKVyKNkZJmXKGW2otkUEFcqUqnsOg` |
| VAPID Private Key | ✅ | `dsrLFPf51vU6KeOPD-Cvd-m7VxiPYHjihIXVUwPYaog` |
| Database Schema | ✅ | Applied to Supabase |
| Mobile App Code | ✅ | Firebase/Notifee integrated |
| Web Push Code | ✅ | VAPID configured |

---

## 🔧 Firebase Setup (Android Push) - ~10 minutes

### Step 1: Create Firebase Project

1. Go to: **https://console.firebase.google.com/**
2. You're logged in as: nicoletterankin@gmail.com
3. Click **"Create a project"** (or "Add project")
4. Project name: `Curious Kelly`
5. Project ID will auto-generate (something like `curious-kelly-12345`)
6. Click **Continue**
7. Disable Google Analytics (we use our own) → **Create project**
8. Wait ~30 seconds for project to create
9. Click **Continue** when done

### Step 2: Add Android App

1. On the project overview, click the **Android icon** (🤖)
2. Package name: `com.curiouskelly.app`
3. App nickname: `Curious Kelly`
4. Debug signing certificate: Leave blank (not needed for FCM)
5. Click **Register app**
6. Click **Download google-services.json**
7. Save this file - I'll tell you where to put it!
8. Click **Continue** through the remaining steps

### Step 3: Get Server Key

1. In Firebase Console, click the **gear icon** ⚙️ → **Project settings**
2. Click **Cloud Messaging** tab
3. Look for "Cloud Messaging API (Legacy)"
4. If it says "Disabled", click the **three dots** → **Enable**
5. Copy the **Server key** (starts with `AAAA...`)
6. **SAVE THIS KEY** - we'll add it to Vercel

### Step 4: Add iOS App (while you're here)

1. Go back to Project Overview
2. Click the **iOS icon** (🍎)
3. Bundle ID: `com.curiouskelly.app`
4. App nickname: `Curious Kelly`
5. Click **Register app**
6. Click **Download GoogleService-Info.plist**
7. Save this file - I'll tell you where to put it!

---

## 🍎 Apple APNs Setup - ~15 minutes

### Step 1: Create APNs Key

1. Go to: **https://developer.apple.com/account/resources/authkeys/list**
2. Log in with your Apple Developer account
3. Click **"+"** to create a new key
4. Name: `Curious Kelly Push`
5. Check **"Apple Push Notifications service (APNs)"**
6. Click **Continue** → **Register**
7. **IMPORTANT**: Click **Download** to get the `.p8` file
   - ⚠️ You can only download this ONCE!
   - Save it somewhere safe!
8. Note these values:
   - **Key ID**: 10-character ID shown (like `ABC123DEFG`)
   - **Team ID**: Your Apple Developer Team ID (found in top right of portal)

### Step 2: Find Your Team ID

1. Look at the top right of Apple Developer portal
2. Or go to **Account** → Look for "Team ID"
3. It's a 10-character alphanumeric code

---

## 📋 What to Give Me

After completing the steps above, share these with me:

### From Firebase:
- [ ] `google-services.json` file (upload it or paste contents)
- [ ] Firebase Server Key (the `AAAA...` string)
- [ ] `GoogleService-Info.plist` file

### From Apple:
- [ ] `.p8` file contents (open in text editor, paste all)
- [ ] Key ID (10 characters)
- [ ] Team ID (10 characters)

---

## 🔐 Security Note

These keys are sensitive but NOT secret like passwords:
- Firebase Server Key can send push notifications to your users
- APNs key can send push notifications to your iOS users
- They should be in environment variables, not committed to git

I will add them to Vercel environment variables (encrypted, never visible in code).

---

## 🚀 After You Provide These

1. I'll add all credentials to Vercel environment variables
2. Deploy the push notification system
3. Test on all platforms
4. We'll be ready for launch! 🎉

---

## Need Help?

If you get stuck on any step, just tell me where you are and I'll help guide you through it!



