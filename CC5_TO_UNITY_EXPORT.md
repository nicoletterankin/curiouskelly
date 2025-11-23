# 🦄 How to Export Kelly from CC5 to Unity

**Goal:** Turn your `.ccProject` into a `.fbx` file that Unity can read.

---

## 1. In Character Creator 5 (CC5)

1.  Open your project (`CC5 Cloth update 1.1.ccProject`).
2.  Go to **File** -> **Export** -> **FBX** -> **Clothed Character**.
3.  Use these **EXACT Settings**:

    *   **Target Tool Preset:** `Unity 3D`
    *   **FBX Options:** `Mesh` (checked)
    *   **Embed Textures:** `Check this box` ✅ (Critical!)
    *   **Delete Hidden Faces:** `Uncheck` (Keep them safe)
    *   **Bake Diffuse Maps:** `Check` (if you want simple textures)

4.  Click **Export**.
5.  Name the file: `Kelly_Real_28.fbx`

---

## 2. What to Give Unity

You need to move that exported file into your new Unity Project.

1.  Find `Kelly_Real_28.fbx` on your computer.
2.  Copy it.
3.  Paste it into: `C:\Users\user\UI-TARS-desktop\digital-kelly\engines\Kelly_Live_2025\Assets\`

*(Note: If you haven't created the `Kelly_Live_2025` project yet, do that first using the Unity Hub).*

---

## 3. What Next?

Once the file is in the folder:
1.  Open Unity.
2.  You will see Kelly appear in the bottom **Project** window.
3.  Drag her into the **Scene** (center window).
4.  She's ready for the web!




