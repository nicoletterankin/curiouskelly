# 🚀 Curious Kelly - Unity Engine Pipeline (Start to Finish)

**Goal:** Prepare the Unity project to receive the FINAL Kelly avatar (sitting in chair) on Nov 25th.
**Current State:** We have a standing avatar (v1). We need to set up the environment, plugins, and lighting so the final import is drag-and-drop.

---

## 🛠️ Phase 1: Unity Project Setup (Do this NOW)

1.  **Create Project:**
    *   Unity Hub -> New Project.
    *   Template: **3D (URP)** (Universal Render Pipeline) - *Critical for hair/eye quality.*
    *   Name: `Kelly_Engine_V2`
    *   Location: `C:\Users\user\UI-TARS-desktop\digital-kelly\engines\`

2.  **Project Settings (Web Ready):**
    *   File -> Build Settings -> Switch Platform to **WebGL**.
    *   Player Settings -> Color Space: **Linear** (Better lighting).
    *   Player Settings -> WebGL -> Compression Format: **Disabled** (Prevents double-zip bug).

---

## 🧩 Phase 2: Plugin Installation (The "Auto Setup")

This is the bridge between Character Creator (CC) and Unity.

1.  **Locate Files:**
    *   I have extracted the plugin to: `C:\Users\user\Downloads\CCIC_Extracted\`
2.  **Install in Unity:**
    *   Copy the folder `Plugins` from the extracted location.
    *   Paste it into `Kelly_Engine_V2/Assets/`.
    *   *Result:* You will see a "Tools" menu appear in Unity.

---

## 💡 Phase 3: The Scene (Prepare the Stage)

Even without the final model, we can build the set.

1.  **Lighting:**
    *   Remove the default "Directional Light" (too harsh).
    *   Add **Area Lights** (Left, Right, Top) to simulate a studio.
    *   Add a **Reflection Probe** (Realtime) to make her eyes sparkle.
2.  **Camera:**
    *   Position Main Camera at `(0, 1.1, 2)` (Head height, slightly back).
    *   Field of View: `35-40` (Portrait lens look).
3.  **Background:**
    *   Create a large curved plane (cyc wall) behind her.
    *   Material: Matte White (or Kelly Brand Blue #4A90E2).

---

## 📦 Phase 4: The Hand-off (Nov 25th Protocol)

When the final files arrive:

1.  **Export from CC/iClone:**
    *   Format: **FBX**.
    *   Target: **Unity 3D**.
    *   Embed Textures: **Checked**.
2.  **Import to Unity:**
    *   Drag FBX into `Assets`.
    *   Auto Setup Popup -> Click **High Quality (URP)**.
3.  **Drag to Scene:**
    *   Drop prefab into the scene.
    *   Align with the Camera we set up in Phase 3.

---

**Status:** Guide Created. Ready to execute Phase 1 (Project Creation).
























