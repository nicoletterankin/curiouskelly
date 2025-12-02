# 🔧 How to Install "Auto Setup" for Unity (The Easy Way)

This plugin is the "Magic Button" that makes Kelly look real (fix eyes, hair, skin) instead of like a plastic toy.

---

## 1. Download the Plugin (Free)
You can't install it from inside Unity. You must grab the file first.

**🔗 Link:** [Click Here to Download from Reallusion](https://www.reallusion.com/auto-setup/unity/download.html)
*   *Note: You might need to log in with your Reallusion account.*
*   **Select:** "Unity Auto Setup" (Look for the version matching Unity 2022/6000).

---

## 2. Install it in Unity
1.  **Unzip** the file you just downloaded.
    *   You will see a folder called `Auto Setup 1.3.x for Unity`.
2.  Open your **Unity Project** (`Kelly_Live_2025`).
3.  Open your File Explorer to the unzipped folder.
4.  **Drag and Drop** the entire `Plugins` (or `CC_Assets`) folder directly into your Unity **Project Window** (bottom panel, `Assets` folder).
    *   *Alternative:* Go to Unity Menu -> **Window** -> **Package Manager** -> **+ (Plus Icon)** -> **Add package from disk** -> Select `package.json` inside the downloaded folder.

---

## 3. Verify It Worked
Look at the top menu bar in Unity.
You should now see a new menu item:
**`Tools` -> `Character Creator & iClone Auto Setup`**

---

## 4. Import Kelly (The Magic Moment)
**Now** drag your `Kelly_Live_v1.fbx` (from iClone) into Unity.
*   Because the plugin is installed, a window will pop up asking "Standard" or "High Quality"?
*   **Select:** `High Quality` (Standard Shader or URP/HDRP depending on your project).
*   **Result:** It will automatically build the materials, fix the transparency on her hair, and make her eyes shine.

---

## 💡 Troubleshooting
*   **"I don't see the menu":** Restart Unity.
*   **"She looks pink":** You didn't enable the Universal Render Pipeline (URP) in the plugin settings. Go to `Tools` -> `Auto Setup` -> `Render Pipeline` -> `URP`.

















