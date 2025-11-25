# 🎬 Unity Fresh Start Guide

**Context:**
You are "starting from scratch" with the new assets (Kelly 28 + Chair).
We need to ensure your new Unity project exports to the exact folder the website expects.

---

## 1. Create the New Project
1.  Open Unity Hub.
2.  Click **New Project**.
3.  Select **3D (URP)** (Universal Render Pipeline) - *Recommended for WebGL visual quality*.
4.  Name it: `Kelly_Live_2025`
5.  Location: `C:\Users\user\UI-TARS-desktop\digital-kelly\engines\`
6.  Click **Create Project**.

---

## 2. Import Your New Assets
1.  Drag your `.fbx` / `.blend` files (Kelly + Chair) into the `Assets` folder.
2.  Drag your Textures/Materials in.
3.  Set up the Scene:
    *   Drag the **Chair** into the scene.
    *   Drag **Kelly** into the chair.
    *   Add a **Camera** positioned for the "Time Magazine" shot.
    *   Add **Lights** (Directional + Soft Area Lights).

---

## 3. WebGL Export Settings (CRITICAL)
*Do this exactly to avoid the "Double Compression" bug.*

1.  Go to **File** -> **Build Settings**.
2.  Select **WebGL**.
3.  Click **Switch Platform** (wait for recompile).
4.  Click **Player Settings** (bottom left).
5.  **Resolution and Presentation:**
    *   WebGL Template: **Default** (or Minimal).
    *   Width: `960` / Height: `600`.
6.  **Publishing Settings (THE FIX):**
    *   **Compression Format:** ⚠️ **Disabled** ⚠️
    *   *(Do NOT set to Gzip or Brotli. Keep it Disabled).*

---

## 4. The Build
1.  In Build Settings, click **Build**.
2.  Navigate to: `C:\Users\user\UI-TARS-desktop\public\unity\`
3.  Create a folder named: `kelly-live`
4.  Select that folder.
5.  **Build!**

---

## 5. Verify
Once built, you should see these files in `public/unity/kelly-live/Build`:
*   `Kelly_Live_2025.data` (or similar)
*   `Kelly_Live_2025.wasm`
*   `Kelly_Live_2025.framework.js`

**(Note: File names might vary based on project name, but extensions must be .data, .wasm, .js)**

---

## 🚀 Next Step
Once the build is done, tell me: *"Build complete."*
I will then wire the `curiouskelly-landing-page.html` to look at this new `kelly-live` folder.











