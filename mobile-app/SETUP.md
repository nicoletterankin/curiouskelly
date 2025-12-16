# React Native Mobile App Setup

## Current Problem

The `mobile-app/` folder contains only the JavaScript code (`App.js`, `index.js`) but is missing the native platform folders (`android/`, `ios/`).

## Solution: Initialize Full React Native Project

### Step 1: Initialize New Project (One-Time)

Run this in a terminal **outside** the current `mobile-app/` folder:

```bash
# Navigate to project root
cd c:\Users\user\UI-TARS-desktop

# Create a new React Native project
npx react-native@0.73.0 init CuriousKellyMobile --version 0.73.0

# This creates:
# CuriousKellyMobile/
# ├── android/          ← Native Android code
# ├── ios/              ← Native iOS code
# ├── App.tsx           ← Default app (we'll replace)
# ├── index.js          ← Entry point
# ├── package.json      ← Dependencies
# └── ...
```

### Step 2: Merge Our Code

After initialization:

```bash
# 1. Backup new project's android/ and ios/
# 2. Copy our App.js over the default App.tsx
# 3. Merge package.json dependencies
# 4. Move android/ and ios/ into mobile-app/
```

### Step 3: Install Dependencies

Our `package.json` requires these native modules that need linking:

```bash
cd mobile-app
npm install

# iOS only (on Mac)
cd ios && pod install && cd ..
```

### Step 4: Configure Firebase

1. Create Firebase project at https://console.firebase.google.com
2. Add iOS app: Bundle ID `com.curiouskelly.mobile`
3. Add Android app: Package `com.curiouskelly.mobile`
4. Download config files:
   - `GoogleService-Info.plist` → `ios/CuriousKelly/`
   - `google-services.json` → `android/app/`

### Step 5: Run

```bash
# iOS (Mac only)
npx react-native run-ios

# Android (Windows/Mac/Linux)
npx react-native run-android
```

## Package.json Dependencies Merge

When merging, ensure these are in the new package.json:

```json
{
  "dependencies": {
    "react": "18.2.0",
    "react-native": "0.73.0",
    "react-native-webview": "^13.6.4",
    "react-native-splash-screen": "^3.3.0",
    "@react-native-async-storage/async-storage": "^1.21.0",
    "react-native-device-info": "^10.12.0",
    "react-native-permissions": "^4.0.3",
    "@react-native-firebase/app": "^18.8.0",
    "@react-native-firebase/messaging": "^18.8.0",
    "@notifee/react-native": "^7.8.0"
  }
}
```

## App Configuration

### iOS: Update Bundle ID

Edit `ios/CuriousKelly/Info.plist`:

```xml
<key>CFBundleIdentifier</key>
<string>com.curiouskelly.mobile</string>
```

### Android: Update Package Name

Edit `android/app/build.gradle`:

```gradle
android {
    defaultConfig {
        applicationId "com.curiouskelly.mobile"
    }
}
```

## Build Commands

### Development

```bash
npm run ios      # iOS simulator
npm run android  # Android emulator
```

### Production

```bash
# iOS: Use Xcode → Product → Archive
# Android:
cd android && ./gradlew assembleRelease
# Output: android/app/build/outputs/apk/release/app-release.apk
```







