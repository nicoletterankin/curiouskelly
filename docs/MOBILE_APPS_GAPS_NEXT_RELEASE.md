# Mobile Apps - Day 1 Pilot Ready

**Created:** December 11, 2025  
**Updated:** December 11, 2025  
**Branch:** `main` (local only, holding until mobile release scheduled)  
**Status:** ✅ Installable shells finalized

---

## ✅ COMPLETED FOR DAY 1 PILOT

| Item | Status |
|------|--------|
| Bundle IDs: `com.curiouskelly.mobile` | ✅ |
| App name: "Curious Kelly" | ✅ |
| Android mipmap icons (all 5 densities) | ✅ |
| iOS AppIcon.appiconset (all sizes) | ✅ |
| Desktop icons in `res/` folder | ✅ |
| Firebase config placeholders | ✅ |
| WebView → `learn.html?day=1` | ✅ |
| Age gate for first-time users | ✅ |
| Icon packs zipped | ✅ |

---

## 📦 ZIPPED ICON PACKS

Located at: `test-output/icon-packs/`

| File | Contents | Size |
|------|----------|------|
| `desktop-icons.zip` | icon.png, icon.ico, entitlements.mac.plist, ICON_GENERATION.md | ~50KB |
| `android-icons.zip` | mipmap-hdpi through mipmap-xxxhdpi | ~200KB |
| `ios-icons.zip` | AppIcon.appiconset with all sizes | ~150KB |

---

## 🛠️ BUILD & INSTALL INSTRUCTIONS

### Android

#### Prerequisites
- Node.js 18+
- Android Studio with SDK
- Java 17

#### Build Steps

```bash
# 1. Navigate to mobile app
cd mobile-app

# 2. Install dependencies
npm install

# 3. Start Metro bundler (in separate terminal)
npm run start

# 4. Build and install debug APK
npm run android

# Or build APK only:
cd android && ./gradlew assembleDebug
# Output: android/app/build/outputs/apk/debug/app-debug.apk
```

#### Production Build (requires signing)

```bash
cd android
./gradlew bundleRelease
# Output: android/app/build/outputs/bundle/release/app-release.aab
```

---

### iOS (requires Mac)

#### Prerequisites
- macOS with Xcode 15+
- CocoaPods
- Apple Developer account

#### Build Steps

```bash
# 1. Navigate to mobile app
cd mobile-app

# 2. Install JS dependencies
npm install

# 3. Install iOS dependencies
cd ios
pod install
cd ..

# 4. Build via Xcode
open ios/CuriousKellyMobile.xcworkspace
# Select target device/simulator → Build (⌘B)
```

---

### Windows (Electron)

#### Prerequisites
- Node.js 18+

#### Build Steps

```bash
# 1. Navigate to desktop app
cd desktop-app

# 2. Install dependencies
npm install

# 3. Run in development
npm run dev

# 4. Build installer
npm run build:win
# Output: dist/Curious Kelly Setup 1.0.0.exe
```

---

### macOS (Electron) - requires Mac

```bash
cd desktop-app
npm install

# Generate .icns first (see res/ICON_GENERATION.md)

npm run build:mac
# Output: dist/Curious Kelly-1.0.0.dmg
```

---

### Linux (Electron)

```bash
cd desktop-app
npm install
npm run build:linux
# Output: dist/Curious Kelly-1.0.0.AppImage
```

---

## 🔶 KNOWN BLOCKERS

### P0 - Required for Store Submission

| Blocker | Platform | Status | Action |
|---------|----------|--------|--------|
| Firebase config | All | ⚠️ Placeholder | Get real keys from Firebase Console |
| Code signing | iOS | ❌ | Create Apple Distribution Certificate |
| Code signing | Android | ❌ | Generate release keystore |
| Code signing | macOS | ❌ | Create Developer ID Certificate |
| Push API | Backend | ❌ | Implement `/api/notifications/*` |

### P1 - Required for Polish

| Blocker | Platform | Status | Action |
|---------|----------|--------|--------|
| Splash screen | iOS | ⚠️ Default | Brand with Kelly |
| Splash screen | Android | ⚠️ Default | Brand with Kelly |
| .icns icon | macOS | ⏳ | Generate on Mac (see `res/ICON_GENERATION.md`) |
| Privacy policy | All stores | ⚠️ | Deploy to curiouskelly.com/privacy |

---

## 📦 SECRETS NEEDED

### Firebase Configuration

| Secret | Where to Get | Where to Put |
|--------|--------------|--------------|
| `google-services.json` | Firebase Console → Project Settings → Android app | `mobile-app/android/app/` |
| `GoogleService-Info.plist` | Firebase Console → Project Settings → iOS app | `mobile-app/ios/CuriousKellyMobile/` |
| FCM Server Key | Firebase Console → Cloud Messaging | Vercel env: `FIREBASE_SERVER_KEY` |

**Detailed instructions:** See `mobile-app/FIREBASE_SETUP.md`

### How to Get Firebase Keys

1. Go to [Firebase Console](https://console.firebase.google.com)
2. Create project: `curious-kelly`
3. Add Android app:
   - Package name: `com.curiouskelly.mobile`
   - Download `google-services.json`
4. Add iOS app:
   - Bundle ID: `com.curiouskelly.mobile`
   - Download `GoogleService-Info.plist`
5. Enable Cloud Messaging in Project Settings

### Code Signing Secrets

| Secret | Platform | Where to Store |
|--------|----------|----------------|
| Release keystore | Android | Local secure storage (NOT git) |
| Distribution cert | iOS | macOS Keychain |
| Developer ID | macOS | macOS Keychain |

---

## 📁 FILES CREATED/UPDATED

### desktop-app/res/ (NEW)
```
├── icon.png              # 512×512 PNG (Linux/all platforms)
├── icon.ico              # Windows multi-size icon
├── entitlements.mac.plist # macOS sandbox entitlements
└── ICON_GENERATION.md    # How to create .icns on Mac
```

### mobile-app/android/app/src/main/res/
```
├── mipmap-mdpi/          # 48×48
├── mipmap-hdpi/          # 72×72
├── mipmap-xhdpi/         # 96×96
├── mipmap-xxhdpi/        # 144×144
└── mipmap-xxxhdpi/       # 192×192
    ├── ic_launcher.png
    └── ic_launcher_round.png
```

### mobile-app/ios/.../AppIcon.appiconset/
```
├── Contents.json
├── icon-40.png           # 40×40
├── icon-60.png           # 60×60
├── icon-87.png           # 87×87
├── icon-120.png          # 120×120
├── icon-180.png          # 180×180
└── icon-1024.png         # 1024×1024 (App Store)
```

### Updated Files
```
mobile-app/App.js                   # WebView → learn.html?day=1
mobile-app/FIREBASE_SETUP.md        # Firebase key documentation
desktop-app/package.json            # buildResources → res/
desktop-app/src/main.js             # Icon path → res/icon.png
```

---

## 🧪 SMOKE TEST RESULTS (December 11, 2025)

### Android Emulator (API 36)

| Test | Result | Notes |
|------|--------|-------|
| Gradle build | ✅ PASS | BUILD SUCCESSFUL in ~4 min |
| APK install | ✅ PASS | Installed on Medium_Phone_API_36.1 |
| App launch | ✅ PASS | MainActivity started |
| Kelly icons | ✅ PASS | Visible in launcher |
| WebView loads | ✅ PASS | Points to learn.html?day=1 |

**Screenshot:** `test-output/android-pilot-screenshot.png`

### Windows Electron

| Test | Result | Notes |
|------|--------|-------|
| npm install | ✅ PASS | Dependencies installed |
| npm run start | ✅ PASS | Window opens, icon visible |
| Kelly icon | ✅ PASS | Visible in taskbar |

### iOS (requires Mac)

| Test | Result | Notes |
|------|--------|-------|
| Build | ⏳ N/A | Requires Mac with Xcode |
| pod install | ⏳ N/A | Requires Mac |

---

## 📋 NEXT SPRINT CHECKLIST

### Day 1: Firebase Setup
- [ ] Create Firebase project `curious-kelly`
- [ ] Register Android app (`com.curiouskelly.mobile`)
- [ ] Register iOS app (`com.curiouskelly.mobile`)
- [ ] Download and replace placeholder config files
- [ ] Enable Cloud Messaging

### Day 1-2: Code Signing
- [ ] Generate Android release keystore
- [ ] Create iOS Distribution Certificate
- [ ] Create iOS Provisioning Profile
- [ ] Run `pod install` on Mac

### Day 2: Backend Push API
- [ ] Create `api/notifications/subscribe.ts`
- [ ] Create `api/notifications/send.ts`
- [ ] Add `firebase-admin` dependency
- [ ] Store Firebase service account in Vercel

### Day 3: Store Prep
- [ ] Capture App Store screenshots (6.7", 6.5", 5.5")
- [ ] Capture Play Store screenshots (phone, tablet)
- [ ] Create Android feature graphic (1024×500)
- [ ] Deploy privacy policy to curiouskelly.com/privacy
- [ ] Submit to TestFlight
- [ ] Submit to Google Play Internal Testing

---

## 🔗 BUILD ARTIFACTS

| Platform | Location | Status |
|----------|----------|--------|
| Android APK | `mobile-app/android/app/build/outputs/apk/debug/` | Build on demand |
| iOS IPA | Xcode → Archive | Requires Mac |
| Windows EXE | `desktop-app/dist/` | Build on demand |
| macOS DMG | `desktop-app/dist/` | Requires Mac |
| Icon packs | `test-output/icon-packs/` | ✅ Ready |

---

## 🚫 BRANCH STATUS

| Item | Status |
|------|--------|
| Branch | `main` (local changes) |
| Push status | **HOLDING** until mobile release scheduled |
| Web deploy | ✅ Unaffected - mobile files are untracked |

---

## 🏁 READY FOR PILOT

When you're ready to proceed:

1. **Get Firebase keys** → Replace placeholder configs
2. **Run on Mac** → `pod install` + iOS build
3. **Generate .icns** → See `desktop-app/res/ICON_GENERATION.md`
4. **Schedule release** → I'll push the branch

**Next action required:** Firebase project creation + config download.


















