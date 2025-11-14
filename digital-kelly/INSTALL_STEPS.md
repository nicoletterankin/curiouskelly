# Kelly OS - Step-by-Step Installation

## 📋 What You Need To Install (In Order)

### 1️⃣ JDK 17
**Purpose:** Required for Android/Gradle builds  
**Download:** https://adoptium.net/temurin/releases/?version=17  
**File:** Windows x64 Installer (.msi)

**After Installation:**
```powershell
java -version  # Should show version 17.x
```

---

### 2️⃣ Flutter SDK
**Purpose:** Cross-platform framework  
**Download:** https://flutter.dev/docs/get-started/install/windows  
**Extract to:** `C:\src\flutter`

**Add to PATH:**
- Search "Environment Variables" in Windows
- Add: `C:\src\flutter\bin` to PATH
- Restart PowerShell

**Verify:**
```powershell
flutter --version
```

---

### 3️⃣ Android Studio
**Purpose:** Android SDK and tools  
**Download:** https://developer.android.com/studio  
**Install:** Standard installer

**After Installation:**
- Open Android Studio
- Tools → SDK Manager
- Install: Platform-Tools, Build-Tools, Command-line Tools
- Accept licenses

**Set ANDROID_HOME:**
```powershell
# Find SDK location (usually)
C:\Users\user\AppData\Local\Android\Sdk

# Set environment variable
[System.Environment]::SetEnvironmentVariable("ANDROID_HOME", "C:\Users\user\AppData\Local\Android\Sdk", "User")
```

---

### 4️⃣ Accept Android Licenses

```powershell
flutter doctor --android-licenses

# Type 'y' for each prompt
```

---

### 5️⃣ Verify Everything Works

```powershell
flutter doctor -v
```

**Target Output:**
```
Doctor summary (all green ✅):
[✓] Flutter (Channel stable, version...)
[✓] Android toolchain
[✓] Android Studio
[✓] VS Code (or your IDE)
```

---

## 🚀 After Installation - Run the App

```powershell
# Navigate to project
cd C:\Users\user\UI-TARS-desktop\digital-kelly\apps\kelly_app_flutter

# Get dependencies
flutter pub get

# Run app
flutter run

# Or build APK
flutter build apk --debug
```

---

## 📞 Need Help?

See detailed guides:
- `SETUP_GUIDE.md` - Complete setup instructions
- `GETTING_STARTED.md` - Quick reference
- `STATUS_REPORT.md` - Current status



















