# Kelly OS - Complete Setup Guide

Follow these steps in order to get your Kelly OS app running.

## ✅ Current Status

- ❌ Java JDK: Not installed
- ❌ Flutter: Not installed  
- ❌ Android SDK: Not installed
- ✅ Audio file: Ready at `C:\Users\user\DigitalKellyTest\audio\kelly_intro.wav`
- ✅ Project: Complete with 45+ files

---

## Step 1: Install JDK 17

### Download & Install

1. **Download JDK 17:**
   - Visit: https://adoptium.net/temurin/releases/?version=17
   - Select: Windows x64 Installer (.msi)
   - Download the file

2. **Install JDK:**
   - Run the downloaded `.msi` file
   - Click "Next" through the installer
   - **IMPORTANT:** Check "Set JAVA_HOME variable"
   - Complete installation

3. **Verify Installation:**
   ```powershell
   java -version
   ```
   Should show: `openjdk version "17.x.x"` or similar

4. **Set JAVA_HOME (if needed):**
   ```powershell
   [System.Environment]::SetEnvironmentVariable("JAVA_HOME", "C:\Program Files\Eclipse Adoptium\jdk-17.x.x-hotspot", "User")
   ```
   Then restart PowerShell.

---

## Step 2: Install Flutter SDK

### Download & Install

1. **Download Flutter:**
   - Visit: https://flutter.dev/docs/get-started/install/windows
   - Click "Download Flutter SDK"
   - Extract to: `C:\src\flutter` (or your preferred location)

2. **Add to PATH:**
   - Search Windows for "Environment Variables"
   - Click "Edit the system environment variables"
   - Click "Environment Variables"
   - Click "Path" → Edit"
   - Click "New"
   - Add: `C:\src\flutter\bin`
   - Click "OK" on all dialogs

3. **Restart PowerShell:**
   - Close and reopen terminal

4. **Verify:**
   ```powershell
   flutter --version
   ```
   Should show Flutter version

---

## Step 3: Install Android Studio (for SDK)

### Download & Install

1. **Download Android Studio:**
   - Visit: https://developer.android.com/studio
   - Click "Download Android Studio"
   - Run installer

2. **Initial Setup:**
   - Launch Android Studio
   - Click "More Actions → SDK Manager"
   - Install:
     - Android SDK Platform-Tools
     - Android SDK Build-Tools  
     - Android SDK Command-line Tools
   - Click "Apply"

3. **Set ANDROID_HOME:**
   - Find your Android SDK location (usually `C:\Users\user\AppData\Local\Android\Sdk`)
   - Add environment variable:
     - Variable name: `ANDROID_HOME`
     - Variable value: `C:\Users\user\AppData\Local\Android\Sdk`

4. **Add to PATH:**
   - Add these to your PATH:
     - `%ANDROID_HOME%\platform-tools`
     - `%ANDROID_HOME%\tools`
     - `%ANDROID_HOME%\cmdline-tools\latest\bin`

---

## Step 4: Accept Android Licenses

```powershell
# Run Flutter doctor to see what's needed
flutter doctor

# Accept licenses
flutter doctor --android-licenses

# Type 'y' for each prompt
```

---

## Step 5: Verify Flutter Doctor

```powershell
flutter doctor -v
```

**Should show:**
- ✅ Flutter: Installed
- ✅ Android toolchain: Installed  
- ✅ Android Studio: Installed
- ✅ VS Code (optional)
- ⚠️  Chromium (optional)
- ⚠️  Windows (optional)

All critical checks should be green!

---

## Step 6: Copy Demo Assets

```powershell
# Navigate to project
cd C:\Users\user\UI-TARS-desktop\digital-kelly

# Copy assets to Documents directory (for Flutter to access)
$docs = (Get-PathFolder -Path Documents).FullName
$targetDir = "$docs\KellyOSAssets"

New-Item -ItemType Directory -Force -Path $targetDir

# Copy A2F JSON
Copy-Item "assets\a2f\kelly_a2f_cache.json" "$targetDir\kelly_a2f_cache.json"

Write-Host "✅ Assets copied to: $targetDir" -ForegroundColor Green
```

---

## Step 7: Get Flutter Dependencies

```powershell
cd apps\kelly_app_flutter

flutter pub get
```

This downloads all Dart packages.

---

## Step 8: Run Flutter App (Development)

```powershell
# See available devices
flutter devices

# Run on Chrome (for testing)
flutter run -d chrome

# OR run on connected Android device
flutter run
```

---

## Step 9: Test "Play Test" Button

When app launches:

1. **Black screen appears**
2. **"Play Test" button** visible (top-right)
3. **Tap button**
4. **Check console for:**
   ```
   📨 Kelly OS: Sent play message to Unity
   📥 KellyBridge: Received load request
   ✅ KellyBridge: Audio playing in sync
   ```

5. **Audio plays** from your `kelly_intro.wav`

---

## Step 10: Build Android Debug APK

```powershell
cd apps\kelly_app_flutter

# Build debug APK
flutter build apk --debug

# APK location:
# build\app\outputs\flutter-apk\app-debug.apk
```

### Install to Device

1. **Connect Android device via USB**
2. **Enable USB Debugging:**
   - Settings → About Phone → Tap "Build Number" 7 times
   - Settings → Developer Options → Enable "USB Debugging"
3. **Verify device connected:**
   ```powershell
   flutter devices
   ```
4. **Install:**
   ```powershell
   flutter install
   ```
   Or manually install: `adb install build\app\outputs\flutter-apk\app-debug.apk`

---

## 🎯 Quick Reference Commands

```powershell
# Check Java
java -version

# Check Flutter
flutter doctor

# Accept licenses
flutter doctor --android-licenses

# Get dependencies
flutter pub get

# Run app
flutter run

# List devices
flutter devices

# Build APK
flutter build apk --debug
```

---

## 🚨 Troubleshooting

### Java not found?
- Install JDK 17 from adoptium.net
- Set JAVA_HOME environment variable
- Restart PowerShell

### Flutter not found?
- Install Flutter SDK
- Add `C:\src\flutter\bin` to PATH
- Restart PowerShell

### Android licenses not accepted?
- Run: `flutter doctor --android-licenses`
- Type 'y' for each prompt

### No devices found?
- Connect Android device via USB
- Enable USB Debugging
- Run: `flutter devices`

### Build fails?
- Run: `flutter clean`
- Run: `flutter pub get`
- Try again: `flutter build apk --debug`

---

## ✅ Success Checklist

- [ ] Java 17+ installed and working
- [ ] Flutter SDK installed
- [ ] Android Studio installed
- [ ] Android SDK installed
- [ ] Licenses accepted
- [ ] `flutter doctor` all green
- [ ] Assets copied to Documents
- [ ] Dependencies installed (`flutter pub get`)
- [ ] App runs (`flutter run`)
- [ ] "Play Test" button works
- [ ] Unity logs appear in console
- [ ] Audio plays successfully
- [ ] APK builds (`flutter build apk --debug`)
- [ ] Installed to device

---

## 🎊 You're Done When...

- `flutter run` launches the app
- Black screen with Unity view appears
- "Play Test" button visible
- Tapping button → Console shows Unity messages
- Audio plays
- APK builds successfully

**Then Kelly will speak! 🎉**


















