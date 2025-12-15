# 📱 MOBILE APP LAUNCH CHECKLIST
## December 17, 2025 Launch — 6 Days Out

**Status:** 🟡 IN PROGRESS - Projects Initialized  
**Owner:** AI Assistant (Mobile App Lead)  
**Last Updated:** December 11, 2025 7:37 AM

---

## 🎯 LAUNCH REQUIREMENTS

| Platform | Build Status | Store Status | Ready |
|----------|--------------|--------------|-------|
| iOS | 🟡 Project Ready | 🔴 Not Submitted | ⚠️ |
| Android | 🟡 Project Ready | 🔴 Not Submitted | ⚠️ |
| Windows | ✅ Tested | N/A (GitHub) | ✅ |
| macOS | 🟡 Needs .icns | N/A (GitHub) | ⚠️ |
| Linux | ✅ Ready | N/A (GitHub) | ✅ |
| Roku | 🟡 Needs Images | 🔴 Not Submitted | ⚠️ |

---

## 📋 PHASE 1: PROJECT INITIALIZATION (Day 1 - Dec 11)

### Mobile App (React Native)

- [x] **1.1** Initialize React Native project ✅ DONE Dec 11
- [x] **1.2** Move initialized project contents to `mobile-app/` ✅ DONE
- [x] **1.3** Merge existing `App.js` (push notifications, WebView) ✅ DONE
- [x] **1.4** Merged dependencies in `package.json` ✅ DONE
- [ ] **1.5** Verify `npx react-native run-ios` works (NEEDS MAC)
- [ ] **1.6** Verify `npx react-native run-android` works

### Desktop App (Electron)

- [x] **1.7** Create `desktop-app/build/` folder ✅ DONE
- [x] **1.8** Generate icon files: ✅ DONE (except .icns)
  - `icon.png` (512×512) — Linux ✅
  - `icon.ico` (256×256) — Windows ✅
  - `icon.icns` — macOS (NEEDS MAC TO GENERATE)
- [x] **1.9** Create `entitlements.mac.plist` ✅ DONE
- [x] **1.10** Run `npm install` ✅ DONE
- [x] **1.11** Verify `npm run dev` launches app ✅ TESTED - WORKS

### Roku App

- [x] **1.12** Create `roku-app/images/` folder ✅ DONE
- [ ] **1.13** Generate channel assets: (NEED TO GENERATE)
  - `icon_focus_hd.png` (540×405)
  - `icon_focus_sd.png` (290×218)
  - `splash_hd.jpg` (1280×720)
  - `splash_sd.jpg` (720×480)
- [ ] **1.14** Verify manifest references resolve

---

## 📋 PHASE 2: ASSETS & BRANDING (Day 1-2)

### App Icons (Master: 1024×1024 PNG)

- [ ] **2.1** Create master app icon (1024×1024)
- [ ] **2.2** Generate iOS icon set (20×20 to 1024×1024, all @1x/@2x/@3x)
- [ ] **2.3** Generate Android icon set (mdpi to xxxhdpi)
- [ ] **2.4** Generate Windows/macOS/Linux icons
- [ ] **2.5** Place icons in correct platform folders

### Splash Screens

- [ ] **2.6** Create iOS launch screen (storyboard or images)
- [ ] **2.7** Create Android splash (drawable resources)
- [ ] **2.8** Create Roku splash screens (HD/SD)

### Screenshots for Store Listings

- [ ] **2.9** iOS screenshots: 6.7" (1290×2796), 6.5" (1284×2778), 5.5" (1242×2208)
- [ ] **2.10** Android screenshots: Phone + Tablet
- [ ] **2.11** Roku screenshots: 1280×720, 1920×1080

---

## 📋 PHASE 3: CONFIGURATION (Day 2-3)

### Firebase Setup (Push Notifications)

- [ ] **3.1** Create Firebase project "curious-kelly"
- [ ] **3.2** Add iOS app (bundle ID: `com.curiouskelly.mobile`)
- [ ] **3.3** Download `GoogleService-Info.plist` → `ios/`
- [ ] **3.4** Add Android app (package: `com.curiouskelly.mobile`)
- [ ] **3.5** Download `google-services.json` → `android/app/`
- [ ] **3.6** Configure APNs for iOS (upload .p8 key)
- [ ] **3.7** Test push notification delivery

### Code Signing

**iOS:**
- [ ] **3.8** Create App ID in Apple Developer Portal
- [ ] **3.9** Create Distribution Certificate
- [ ] **3.10** Create Provisioning Profile (App Store)
- [ ] **3.11** Configure Xcode signing

**Android:**
- [ ] **3.12** Generate release keystore
  ```bash
  keytool -genkey -v -keystore curious-kelly.keystore -alias curious-kelly -keyalg RSA -keysize 2048 -validity 10000
  ```
- [ ] **3.13** Configure `android/gradle.properties` with keystore
- [ ] **3.14** Configure `android/app/build.gradle` signing config

**macOS (Electron):**
- [ ] **3.15** Obtain Developer ID Application certificate
- [ ] **3.16** Configure notarization credentials

---

## 📋 PHASE 4: BUILD & TEST (Day 3-4)

### iOS Build

- [ ] **4.1** Build debug on simulator
- [ ] **4.2** Build on physical device
- [ ] **4.3** Archive release build
- [ ] **4.4** Validate archive (no errors)

### Android Build

- [ ] **4.5** Build debug APK
- [ ] **4.6** Test on emulator
- [ ] **4.7** Build release APK
- [ ] **4.8** Build release AAB (for Play Store)
- [ ] **4.9** Test release APK on device

### Desktop Build

- [ ] **4.10** Build Windows installer
- [ ] **4.11** Build macOS DMG
- [ ] **4.12** Build Linux AppImage
- [ ] **4.13** Test auto-updater

### Roku Build

- [ ] **4.14** Package channel as .zip
- [ ] **4.15** Side-load to Roku device
- [ ] **4.16** Test on physical Roku

---

## 📋 PHASE 5: STORE SUBMISSION (Day 4-5)

### App Store Connect (iOS)

- [ ] **5.1** Create app in App Store Connect
- [ ] **5.2** Fill in app information:
  - Name: Curious Kelly
  - Subtitle: Your AI Teacher for Life
  - Category: Education
  - Age Rating: 4+
- [ ] **5.3** Upload screenshots
- [ ] **5.4** Write description (see template below)
- [ ] **5.5** Set keywords
- [ ] **5.6** Add privacy policy URL
- [ ] **5.7** Add support URL
- [ ] **5.8** Upload build via Xcode/Transporter
- [ ] **5.9** Submit for review
- [ ] **5.10** Request expedited review (holiday launch)

### Google Play Console (Android)

- [ ] **5.11** Create app in Play Console
- [ ] **5.12** Complete app details:
  - App name: Curious Kelly
  - Short description
  - Full description
- [ ] **5.13** Upload feature graphic (1024×500)
- [ ] **5.14** Upload screenshots
- [ ] **5.15** Complete content rating questionnaire
- [ ] **5.16** Set up pricing (Free with IAP)
- [ ] **5.17** Add privacy policy URL
- [ ] **5.18** Upload AAB
- [ ] **5.19** Submit for review

### Roku Channel Store

- [ ] **5.20** Create channel in Roku Developer Portal
- [ ] **5.21** Upload channel package
- [ ] **5.22** Fill in channel properties
- [ ] **5.23** Upload required assets
- [ ] **5.24** Submit for certification

### GitHub Releases (Desktop)

- [ ] **5.25** Create GitHub repository (curiouskelly/desktop-app)
- [ ] **5.26** Upload Windows installer
- [ ] **5.27** Upload macOS DMG
- [ ] **5.28** Upload Linux AppImage
- [ ] **5.29** Create release notes

---

## 📋 PHASE 6: LAUNCH (Day 6 - Dec 17)

- [ ] **6.1** Verify iOS app approved
- [ ] **6.2** Verify Android app approved
- [ ] **6.3** Release iOS to App Store
- [ ] **6.4** Release Android to Play Store
- [ ] **6.5** Announce desktop downloads
- [ ] **6.6** Update website with download links
- [ ] **6.7** Monitor crash reports
- [ ] **6.8** Monitor user reviews

---

## 📝 STORE LISTING TEMPLATES

### App Name
```
Curious Kelly - Daily AI Lessons
```

### Subtitle (iOS) / Short Description (Android)
```
Your personal AI teacher. Learn something new every day.
```

### Description
```
Meet Kelly — your personal AI teacher who delivers a fresh lesson every single day.

🎓 365 DAILY LESSONS
Every day brings a new topic: science, history, creativity, wisdom, and wonder. From "The Sun" on January 1st to reflections on December 31st — a full year of learning awaits.

👩‍🏫 KELLY ADAPTS TO YOU
Whether you're 6 or 60, Kelly adjusts her teaching style to match your age and experience. The same lesson, personalized for you.

📅 BEAUTIFUL CALENDAR
See your entire year of learning. Track your progress. Build your streak. Never miss a lesson.

🔔 GENTLE REMINDERS
Kelly reminds you each morning. Start your day with curiosity.

✨ FEATURES
• Daily live lessons with your AI teacher
• Full 365-day curriculum
• Age-adaptive teaching (2-102)
• Progress tracking and streaks
• Offline access to completed lessons
• Family sharing available

Start your journey with Kelly today. Your first lesson is waiting.
```

### Keywords (iOS)
```
learning,education,AI,teacher,daily,lessons,curriculum,knowledge,wisdom,kids,adults
```

### Privacy Policy URL
```
https://curiouskelly.com/privacy
```

### Support URL
```
https://curiouskelly.com/support
```

---

## 🔐 CREDENTIALS NEEDED (Session with User)

When we work together in browser, I'll need:

### Apple Developer
- [ ] Apple ID email
- [ ] Team ID (10-character)
- [ ] Access to https://developer.apple.com
- [ ] Access to https://appstoreconnect.apple.com

### Google Play
- [ ] Google account email
- [ ] Access to https://play.google.com/console

### Firebase
- [ ] Google account for Firebase Console
- [ ] Or existing Firebase project credentials

### Roku
- [ ] Roku developer account email
- [ ] Access to https://developer.roku.com

### GitHub
- [ ] GitHub organization access
- [ ] Personal access token for releases

---

## 📊 TIMELINE

| Day | Date | Focus |
|-----|------|-------|
| 1 | Dec 11 | Project initialization, assets |
| 2 | Dec 12 | Firebase, icons, configuration |
| 3 | Dec 13 | Build iOS + Android |
| 4 | Dec 14 | Submit to stores |
| 5 | Dec 15 | Desktop builds, Roku |
| 6 | Dec 16 | Buffer / review responses |
| 7 | Dec 17 | 🚀 LAUNCH |

---

## ⚠️ RISKS

| Risk | Mitigation |
|------|------------|
| App Store review delay (1-3 days) | Submit by Dec 14, request expedited review |
| Rejection for metadata | Pre-validate all assets, clear descriptions |
| Push notification issues | Test thoroughly before submission |
| Code signing problems | Set up certificates early |

---

## 📞 SUPPORT

**App Store Review Issues:**  
Apple Developer Support: 1-800-633-2152

**Play Store Issues:**  
https://support.google.com/googleplay/android-developer

---

**Next Step:** Initialize React Native project and create assets.




