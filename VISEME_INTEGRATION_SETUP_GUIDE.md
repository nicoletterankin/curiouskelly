# Unity Avatar Lip-Sync Integration - Complete Setup Guide

Step-by-step guide to build Unity project, connect Flutter app to backend, and stream OpenAI Realtime API viseme data.

---

## Part 1: Unity Project Build & Run

### Prerequisites

1. **Install Unity Hub** (if not already installed)
   - Download: https://unity.com/download
   - Install Unity **2022.3 LTS** or **2023.x**

2. **Verify Unity Installation**
   - Open Unity Hub
   - Check installed versions
   - Unity 2022.3 LTS or later required

### Step 1: Open Unity Project

1. **Open Unity Hub**
   - Click "Projects" tab (left side)
   - Click "Add" button (top right)
   - Navigate to: `C:\Users\user\UI-TARS-desktop\digital-kelly\engines\kelly_unity_player`
   - Click "Select Folder"

2. **Select Unity Version**
   - Unity Hub will show project
   - If version mismatch, click "Open" and Unity will prompt to upgrade
   - **Important:** Choose Unity **2022.3 LTS** or later

3. **Wait for Project Load**
   - Unity will import assets (first time: 5-10 minutes)
   - Wait until bottom-right progress bar completes
   - Console should show no errors (yellow warnings OK)

### Step 2: Configure Build Settings

1. **Open Build Settings**
   - Menu: **File → Build Settings** (or `Ctrl+Shift+B`)
   - Platform list appears

2. **Select Android Platform**
   - In platform list, click **"Android"**
   - Click **"Switch Platform"** button (bottom right)
   - Wait for platform switch (2-5 minutes)

3. **Configure Player Settings**
   - Click **"Player Settings"** button (bottom left)
   - In Inspector panel (right side):
     
     **General:**
     - Company Name: `UI-TARS`
     - Product Name: `Curious Kellly`
     - Version: `0.2.0`

     **Configuration:**
     - Scripting Backend: **IL2CPP**
     - API Compatibility Level: **.NET Framework**
     - Target Architectures: **ARM64** ✓ (uncheck ARMv7)

     **Rendering:**
     - Color Space: **Linear**
     - Target Frame Rate: **60**

     **Other Settings:**
     - Minimum API Level: **API Level 24** (Android 7.0)
     - Target API Level: **API Level 33+**
     - Graphics APIs: **Vulkan** (first), OpenGLES3 (fallback)

4. **Close Player Settings**
   - Click X on Player Settings window

### Step 3: Verify Scene Setup

1. **Open Main Scene**
   - In Project window (left), navigate: `Assets/Kelly/Scenes/`
   - Double-click `Main.unity` (or `KellyAvatar.unity` if exists)

2. **Verify Scene Contains:**
   - Kelly Avatar GameObject (in Hierarchy)
   - `KellyAvatarController` component attached
   - `BlendshapeDriver60fps` component attached
   - `UnityMessageManager` GameObject

3. **If Missing Components:**
   - Select Kelly Avatar GameObject
   - Inspector panel (right side)
   - Click **"Add Component"**
   - Search and add:
     - `KellyAvatarController`
     - `BlendshapeDriver60fps`
     - `AvatarPerformanceMonitor` (optional)

### Step 4: Test in Unity Editor

1. **Open Console Window**
   - Menu: **Window → General → Console** (or `Ctrl+Shift+C`)
   - Clear console: Click trash icon

2. **Press Play Button** (top center, ▶️)
   - Unity will enter Play Mode
   - Check Console for: `[KellyController] Avatar initialized and ready`
   - Check Console for: `[Kelly60fps] Indexed X blendshapes`

3. **Test Viseme Handler** (Optional)
   - In Hierarchy, select Kelly Avatar
   - Inspector panel → `KellyAvatarController`
   - In OnGUI test area (if visible), Unity will show test buttons

4. **Stop Play Mode**
   - Click Play button again (▶️ becomes ⏸️)

### Step 5: Build for Android

1. **Open Build Settings**
   - Menu: **File → Build Settings**
   - Platform: **Android** (should be selected)

2. **Add Scenes**
   - In Scenes In Build list, ensure `Main.unity` is listed
   - If empty, drag `Main.unity` from Project window into list

3. **Build Settings:**
   - **Build App Bundle (Google Play)**: Unchecked
   - **Development Build**: Checked ✓
   - **Script Debugging**: Checked ✓

4. **Click "Build" Button**
   - Choose output folder: `C:\Users\user\UI-TARS-desktop\curious-kellly\mobile\android\UnityExport`
   - Click "Select Folder"
   - Unity builds (5-10 minutes)

5. **Build Complete:**
   - Unity exports `.aar` library
   - Output: `android\unityLibrary\`
   - Check for no errors in Console

### Step 6: Alternative - Test in Unity Editor Only

**If you don't need Android build yet:**

1. **Open Test Scene**
   - Create new scene: **File → New Scene**
   - Add Kelly Avatar: Drag from Project → Hierarchy

2. **Attach Scripts**
   - Select Avatar
   - Inspector → Add Component → Search "KellyAvatarController"

3. **Play in Editor**
   - Press Play
   - Check Console logs
   - Viseme integration will work in Play Mode

---

## Part 2: Backend Setup & Connection

### Prerequisites

1. **Node.js Installed**
   - Verify: Open PowerShell, run `node --version`
   - Should show v18.x or v20.x
   - If missing: https://nodejs.org/

2. **Backend Dependencies Installed**
   - Navigate: `C:\Users\user\UI-TARS-desktop\curious-kellly\backend`
   - Run: `npm install`

### Step 1: Configure Backend Environment

1. **Create `.env` File**
   - Navigate: `C:\Users\user\UI-TARS-desktop\curious-kellly\backend`
   - Create file: `.env` (if not exists)

2. **Add Environment Variables**
   - Open `.env` in text editor
   - Add these lines:
     ```
     OPENAI_API_KEY=sk-proj-your-key-here
     OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview-2024-10-01
     PORT=3000
     NODE_ENV=development
     ```
   - Replace `sk-proj-your-key-here` with your actual OpenAI API key
   - Save file

3. **Verify `.env` Location**
   - File should be at: `curious-kellly/backend/.env`
   - **Important:** `.env` is gitignored (don't commit)

### Step 2: Install Backend Dependencies

1. **Open PowerShell**
   - Navigate to backend folder:
     ```powershell
     cd C:\Users\user\UI-TARS-desktop\curious-kellly\backend
     ```

2. **Install Packages**
   ```powershell
   npm install
   ```
   - Wait for installation (1-2 minutes)
   - Should show: `added XXX packages`

### Step 3: Start Backend Server

1. **Start Development Server**
   ```powershell
   npm run dev
   ```
   
   **OR** for production:
   ```powershell
   npm start
   ```

2. **Verify Server Started**
   - Console should show:
     ```
     🚀 Curious Kellly Backend Started
     Server running on http://localhost:3000
     ```
   - **Keep this terminal open** (don't close)

3. **Test Backend Health**
   - Open browser: `http://localhost:3000/health`
   - Should show JSON: `{"status":"ok"}`
   - Or test API: `http://localhost:3000/api/realtime/test`

### Step 4: Test WebSocket Connection

1. **Install wscat (WebSocket Client)**
   ```powershell
   npm install -g wscat
   ```

2. **Connect to WebSocket**
   ```powershell
   wscat -c "ws://localhost:3000/api/realtime/ws?learnerAge=35"
   ```

3. **Send Test Message**
   - In wscat terminal, type:
     ```json
     {"type":"offer","sdp":"test-offer"}
     ```
   - Press Enter
   - Should receive response from backend

4. **Close wscat**
   - Press `Ctrl+C`

---

## Part 3: Flutter App Connection

### Prerequisites

1. **Flutter Installed**
   - Verify: `flutter --version`
   - Should show Flutter 3.24+

2. **Flutter Dependencies**
   - Navigate: `C:\Users\user\UI-TARS-desktop\curious-kellly\mobile`
   - Run: `flutter pub get`

### Step 1: Configure Backend URL

1. **Find Config File**
   - Navigate: `curious-kellly/mobile/lib/config/` (or similar)
   - Look for files: `app_config.dart`, `environment.dart`, or check `VoiceController` constructor

2. **Set Backend URL**
   - Open file that contains `backendUrl`
   - Look for line like: `const String backendUrl = ...`
   - Change to:
     ```dart
     const String backendUrl = 'http://localhost:3000';
     ```
   - **For Android Emulator:** Use `http://10.0.2.2:3000`
   - **For iOS Simulator:** Use `http://localhost:3000`
   - **For Physical Device:** Use your computer's IP: `http://192.168.1.X:3000`

3. **If Config Not Found:**
   - Check `lib/controllers/voice_controller.dart`
   - Constructor should accept `backendUrl` parameter
   - Update call sites to pass: `backendUrl: 'http://localhost:3000'`

### Step 2: Configure for Your Device

**Option A: Android Emulator**
```dart
const String backendUrl = 'http://10.0.2.2:3000';
```

**Option B: iOS Simulator**
```dart
const String backendUrl = 'http://localhost:3000';
```

**Option C: Physical Device**
1. Find your computer's IP address:
   ```powershell
   ipconfig
   ```
   Look for "IPv4 Address" (e.g., `192.168.1.105`)
2. Use that IP:
   ```dart
   const String backendUrl = 'http://192.168.1.105:3000';
   ```
3. **Important:** Ensure phone and computer on same WiFi network

### Step 3: Get Flutter Dependencies

1. **Open PowerShell**
   ```powershell
   cd C:\Users\user\UI-TARS-desktop\curious-kellly\mobile
   ```

2. **Install Packages**
   ```powershell
   flutter pub get
   ```
   - Wait for completion
   - Should show: `Got dependencies!`

### Step 4: Connect Unity Build (If Built)

**If you built Unity for Android:**

1. **Verify Unity Export**
   - Check folder exists: `curious-kellly/mobile/android/unityLibrary/`
   - Should contain `.aar` files

2. **Update Android Gradle** (if needed)
   - File: `android/settings.gradle`
   - Ensure includes:
     ```gradle
     include ':unityLibrary'
     project(':unityLibrary').projectDir = new File('unityLibrary')
     ```

3. **Rebuild Flutter**
   ```powershell
   flutter clean
   flutter pub get
   flutter build apk --debug
   ```

### Step 5: Run Flutter App

1. **Check Connected Devices**
   ```powershell
   flutter devices
   ```
   - Should list your device/emulator

2. **Run App**
   ```powershell
   flutter run
   ```
   - Flutter builds and installs
   - App launches on device

3. **Verify Connection**
   - Navigate to Conversation Screen
   - Tap "Connect" button
   - Check status indicator (should turn green)
   - Check logs: Should see WebSocket connection message

---

## Part 4: OpenAI Realtime API & Viseme Streaming

### Prerequisites

1. **OpenAI API Key**
   - Must have access to **Realtime API** (beta)
   - Key format: `sk-proj-...`
   - Verify: https://platform.openai.com/api-keys

2. **Backend Running**
   - Backend server must be running (Part 2, Step 3)
   - `.env` must contain `OPENAI_API_KEY`

### Step 1: Verify OpenAI API Key

1. **Test API Key**
   ```powershell
   # In backend folder
   curl -H "Authorization: Bearer YOUR_API_KEY" https://api.openai.com/v1/models
   ```
   - Should return list of models (not 401 error)

2. **Check Realtime API Access**
   - Log into: https://platform.openai.com/
   - Navigate: **API → Realtime API**
   - Verify access granted (beta feature)

### Step 2: Verify Backend Realtime Service

1. **Check Backend Logs**
   - Backend terminal should show no errors
   - On startup, should see: `OpenAI client initialized`

2. **Test Realtime Endpoint**
   ```powershell
   curl -X POST http://localhost:3000/api/realtime/ephemeral-key -H "Content-Type: application/json" -d "{\"learnerAge\":35}"
   ```
   - Should return JSON with `sessionId` and config
   - **If error:** Check `.env` has correct `OPENAI_API_KEY`

### Step 3: Test Viseme Streaming

1. **Start Flutter App** (from Part 3, Step 5)
   - App should be running
   - Conversation screen open

2. **Connect Voice**
   - Tap "Connect" button
   - Wait for connection (2-3 seconds)
   - Status should show "Connected"

3. **Start Conversation**
   - Tap voice control button (center bottom)
   - Speak: "Hello Kelly"
   - Wait for response

4. **Check Viseme Stream**
   - **Flutter Logs:** Look for:
     ```
     [OpenAIRealtimeService] Visemes received: {...}
     [VoiceController] Visemes received
     ```
   - **Unity Logs (if in Play Mode):** Look for:
     ```
     [KellyController] Parsed X visemes from message
     [Kelly60fps] UpdateVisemes called
     ```
   - **Avatar Should:** Mouth move in sync with Kelly's voice

### Step 4: Debug Viseme Issues

**If visemes not appearing:**

1. **Check Flutter → Unity Bridge**
   - Flutter logs: `[FlutterUnityBridge] Sending visemes`
   - Unity logs: `[UnityMessageManager] Received from Flutter`

2. **Check JSON Parsing**
   - Unity logs should show: `[KellyController] Parsed X visemes`
   - If shows "Could not find visemes object", JSON format issue

3. **Check Blendshape Mapping**
   - Unity logs: `[Kelly60fps] Indexed X blendshapes`
   - If viseme received but no blendshape change, mapping issue

4. **Verify Blendshape Names**
   - In Unity: Select Avatar → Inspector
   - Check `BlendshapeDriver60fps` → `headRenderer`
   - Verify blendshape names match mapping (e.g., `mouthOpen`, `jawOpen`)

### Step 5: Verify Lip-Sync Accuracy

1. **Test Different Phonemes**
   - Say: "Apple" (tests "aa", "PP")
   - Say: "Bee" (tests "ee")
   - Say: "Ice" (tests "ih")
   - Say: "Oh" (tests "oh")
   - Say: "You" (tests "ou")

2. **Observe Avatar Mouth**
   - Should match phoneme sounds
   - Transitions should be smooth (60fps)
   - No jittery movements

3. **Check Performance**
   - Unity Performance Monitor: Should show 60fps
   - If below 55fps, check device specs

---

## Troubleshooting

### Unity Build Fails

**Error: "Unable to convert classes"**
- Solution: Change Scripting Backend to IL2CPP (Part 1, Step 2.3)

**Error: "BlendshapeDriver60fps not found"**
- Solution: Ensure scripts are in `Assets/Kelly/Scripts/` folder

**Error: "UnityMessageManager not found"**
- Solution: Add `UnityMessageManager` GameObject to scene

---

### Backend Won't Start

**Error: "OPENAI_API_KEY not found"**
- Solution: Create `.env` file with API key (Part 2, Step 1)

**Error: "Port 3000 already in use"**
- Solution: 
  ```powershell
   # Find process using port 3000
   netstat -ano | findstr :3000
   # Kill process (replace PID)
   taskkill /PID <PID> /F
   ```

**Error: "Module not found"**
- Solution: Run `npm install` in backend folder

---

### Flutter Can't Connect

**Error: "Connection refused"**
- Solution: Check backend is running (`http://localhost:3000/health`)
- Solution: Verify backend URL in Flutter config matches device type

**Error: "WebSocket connection failed"**
- Solution: Check backend WebSocket route: `ws://localhost:3000/api/realtime/ws`
- Solution: Verify `express-ws` installed: `npm list express-ws`

**Error: "No devices found"**
- Solution: Run `flutter devices` to list available devices
- Solution: For Android, enable USB debugging
- Solution: For iOS, trust computer on device

---

### Visemes Not Streaming

**No visemes in Flutter logs:**
- Solution: Check OpenAI Realtime API access (beta required)
- Solution: Verify API key in `.env` is correct
- Solution: Check backend logs for OpenAI API errors

**Visemes in Flutter but not Unity:**
- Solution: Verify Unity bridge passed to VoiceController (conversation_screen.dart)
- Solution: Check Unity logs for message receipt
- Solution: Verify `updateVisemes` method called in FlutterUnityBridge

**Visemes received but no mouth movement:**
- Solution: Check blendshape names match mapping (Part 4, Step 4)
- Solution: Verify `headRenderer` assigned in BlendshapeDriver60fps
- Solution: Check Unity Console for blendshape errors

---

## Success Criteria

✅ **Unity Project:**
- Builds without errors
- Plays in Unity Editor
- Avatar visible with blendshape driver

✅ **Backend:**
- Starts on port 3000
- Responds to `/health` endpoint
- WebSocket accepts connections

✅ **Flutter App:**
- Connects to backend
- WebSocket established
- Voice conversation works

✅ **Viseme Streaming:**
- Visemes received in Flutter
- Visemes sent to Unity
- Avatar mouth moves in sync
- Smooth 60fps animation

---

## Next Steps After Setup

1. **Test Full Conversation Flow**
   - Multi-turn conversation
   - Barge-in functionality
   - Session persistence

2. **Performance Tuning**
   - Measure latency (target: <100ms viseme→blendshape)
   - Optimize blendshape interpolation
   - Test on multiple devices

3. **Polish & Production**
   - Add error recovery
   - Improve visual quality
   - Add audio-visual sync verification

---

**Ready to test! Start with Part 1 and work through each section sequentially.**













