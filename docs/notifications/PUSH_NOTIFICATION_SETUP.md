# 🔔 Push Notification Setup Guide

> Complete setup guide for Curious Kelly push notifications across all platforms.

**Launch Date**: December 17, 2025

---

## 📋 Overview

| Platform | Technology | Status |
|----------|------------|--------|
| Web | Web Push (VAPID) | ✅ Ready |
| iOS | APNs | 🔧 Needs certificates |
| Android | Firebase Cloud Messaging | 🔧 Needs Firebase project |
| Desktop | Electron + Web Push | ✅ Ready |
| Roku | Deep linking (no push) | ⏳ Future |

---

## 🌐 Web Push (VAPID) — COMPLETE

### Keys Generated
```
VAPID_PUBLIC_KEY=BEgmu91QD3hye9UZ9MM6xZxfbIRrmhiKE3cV3XkfvxAlRMATdRY4skdaFAMKVyKNkZJmXKGW2otkUEFcqUqnsOg
VAPID_PRIVATE_KEY=dsrLFPf51vU6KeOPD-Cvd-m7VxiPYHjihIXVUwPYaog
VAPID_SUBJECT=mailto:hello@curiouskelly.com
```

### Add to Vercel Environment Variables

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Select the `curiouskelly` project
3. Go to Settings → Environment Variables
4. Add these variables for **Production**:

| Name | Value |
|------|-------|
| `VAPID_PUBLIC_KEY` | `BEgmu91QD3hye9UZ9MM6xZxfbIRrmhiKE3cV3XkfvxAlRMATdRY4skdaFAMKVyKNkZJmXKGW2otkUEFcqUqnsOg` |
| `VAPID_PRIVATE_KEY` | `dsrLFPf51vU6KeOPD-Cvd-m7VxiPYHjihIXVUwPYaog` |
| `VAPID_SUBJECT` | `mailto:hello@curiouskelly.com` |

---

## 🍎 iOS APNs Setup

### Prerequisites
- Apple Developer account (you have this!)
- App registered with Bundle ID: `com.curiouskelly.app`

### Step 1: Create APNs Key

1. Go to [Apple Developer Portal](https://developer.apple.com/account/resources/authkeys/list)
2. Click "+" to create a new key
3. Name: `Curious Kelly Push Key`
4. Check "Apple Push Notifications service (APNs)"
5. Click "Continue" → "Register"
6. **IMPORTANT**: Download the `.p8` file (you can only download once!)
7. Note the **Key ID** (10 characters)
8. Note your **Team ID** from your Apple Developer account

### Step 2: Add to Vercel Environment Variables

| Name | Value |
|------|-------|
| `APNS_KEY_ID` | Your Key ID (e.g., `ABC123DEFG`) |
| `APNS_TEAM_ID` | Your Team ID (e.g., `XYZ9876543`) |
| `APNS_AUTH_KEY` | Contents of the `.p8` file (entire key including headers) |
| `APNS_BUNDLE_ID` | `com.curiouskelly.app` |

### Step 3: iOS App Configuration

The mobile app is already configured with Firebase Messaging which handles APNs automatically. Just ensure:

1. `GoogleService-Info.plist` is in `ios/CuriousKelly/`
2. Push Notifications capability is enabled in Xcode
3. Background Modes → Remote notifications is checked

---

## 🤖 Android FCM Setup

### Step 1: Create Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Add Project"
3. Name: `Curious Kelly`
4. Enable Google Analytics (optional)
5. Create project

### Step 2: Add Android App

1. Click "Add app" → Android icon
2. Package name: `com.curiouskelly.app`
3. App nickname: `Curious Kelly`
4. Download `google-services.json`
5. Place it in `mobile-app/android/app/`

### Step 3: Get Server Key (Legacy - Simple Method)

1. In Firebase Console → Project Settings → Cloud Messaging
2. Find "Cloud Messaging API (Legacy)" 
3. If disabled, click "Enable"
4. Copy the **Server key**

### Step 4: Add to Vercel Environment Variables

**Option A: Legacy Server Key (Simpler)**
| Name | Value |
|------|-------|
| `FIREBASE_SERVER_KEY` | Your server key (starts with `AAAA...`) |

**Option B: Service Account (More Secure)**
| Name | Value |
|------|-------|
| `FIREBASE_PROJECT_ID` | Your project ID |
| `FIREBASE_PRIVATE_KEY` | From service account JSON |
| `FIREBASE_CLIENT_EMAIL` | From service account JSON |

To get service account credentials:
1. Firebase Console → Project Settings → Service Accounts
2. Generate new private key
3. Download JSON file
4. Extract `project_id`, `private_key`, `client_email`

---

## 🖥️ Desktop (Electron) — COMPLETE

Desktop apps use Web Push via Electron's built-in notification system. No additional setup required beyond the VAPID keys.

---

## ✅ Verification Checklist

### Vercel Environment Variables

Run this checklist after adding all environment variables:

```
□ VAPID_PUBLIC_KEY
□ VAPID_PRIVATE_KEY
□ VAPID_SUBJECT
□ APNS_KEY_ID
□ APNS_TEAM_ID
□ APNS_AUTH_KEY
□ APNS_BUNDLE_ID
□ FIREBASE_SERVER_KEY (or FIREBASE_PROJECT_ID + FIREBASE_PRIVATE_KEY + FIREBASE_CLIENT_EMAIL)
```

### Test Push Notifications

1. **Web Push Test**
   ```bash
   # In browser console at curiouskelly.com:
   PushNotifications.subscribe().then(console.log)
   ```

2. **Mobile Test**
   - Install app on device (not simulator)
   - Accept notification permission
   - Check Supabase `push_tokens` table for new entry
   - Trigger test notification from dashboard

---

## 🚨 Security Notes

- **NEVER** commit `.p8` files, `google-services.json`, or private keys to git
- All keys should only exist in Vercel environment variables
- Rotate VAPID keys annually (or if compromised)
- APNs keys don't expire but can be revoked
- FCM server keys should be rotated if exposed

---

## 📁 Files Reference

| File | Purpose |
|------|---------|
| `lib/push-sender.ts` | Server-side push sending library |
| `api/notifications/web-push-subscribe.ts` | Web push subscription endpoint |
| `api/notifications/subscribe-device.ts` | Authenticated device registration |
| `api/cron/daily-push-notifications.ts` | Hourly push notification cron |
| `public/js/push-notifications.js` | Web push client |
| `public/sw.js` | Service worker for web push |
| `mobile-app/App.js` | React Native app with Firebase |
| `mobile-app/android/app/google-services.json` | Firebase Android config |
| `mobile-app/ios/CuriousKelly/GoogleService-Info.plist` | Firebase iOS config |

---

## 🆘 Troubleshooting

### Web Push Not Working
1. Check browser console for errors
2. Verify VAPID keys are correct in both client and server
3. Ensure service worker is registered (`navigator.serviceWorker.ready`)
4. Check `push_tokens` table for subscription

### iOS Notifications Not Arriving
1. Verify APNs certificate is valid
2. Check device token format (hex string, no spaces)
3. Use production APNs endpoint for App Store builds
4. Check `notification_log` table for errors

### Android Notifications Not Arriving
1. Verify `google-services.json` is in correct location
2. Check FCM token format
3. Verify server key permissions in Firebase
4. Check `notification_log` table for errors

---

## 📞 Need Help?

If you encounter issues during setup, the critical values I need from you are:

1. **For iOS**: The `.p8` file contents, Key ID, and Team ID
2. **For Android**: The Firebase Server Key (or service account JSON)

I can add these to Vercel and complete the setup.

