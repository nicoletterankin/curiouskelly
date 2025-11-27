# 🧙‍♂️ The "Digital Person" Pipeline: CC5 → iClone → Unity

**Goal:** Export a living, breathing Kelly that can speak (Visemes) and act (Motion) in the browser.

---

## Phase 1: The Soul (iClone 8)
*Don't just export a statue. Export a performance.*

1.  **Send from CC5:** `File` -> `Export` -> `Send Character to iClone`.
2.  **Apply Chair Pose:** Load your Director's Chair pose.
3.  **AccuLips Prep (CRITICAL):**
    *   Go to **Animation** -> **AccuLips**.
    *   Just opening this panel initializes the "Viseme" blendshapes on the mesh.
    *   *Optional:* Drop in a 5-second test audio just to see her talk.
4.  **Facial Profile:**
    *   Go to **Modify** -> **Face Key** -> **Expression Plus**.
    *   Click "Default" to ensure all 63+ ARKit blendshapes are active.
5.  **Idle Loop:**
    *   Add a "Breathing" motion (Motion -> Idle -> Female -> Breathing).
    *   Set timeline to 0-600 frames (10 seconds).

---

## Phase 2: The Package (Export FBX)
1.  **File** -> **Export** -> **FBX**.
2.  **Target:** `Unity 3D`.
3.  **Range:** `Range` (Export the breathing loop).
4.  **Settings:**
    *   ✅ Embed Textures
    *   ✅ Mesh and Motion
    *   ❌ Delete Hidden Faces (Keep them!)
5.  **Name:** `Kelly_Live_v1.fbx`

---

## Phase 3: The Magic (Unity + Auto Setup)
1.  **Install Auto Setup:** Follow `INSTALL_AUTO_SETUP.md`.
2.  **Import:** Drag `Kelly_Live_v1.fbx` into Unity `Assets`.
3.  **Auto-Fix:**
    *   When the popup appears, choose **High Quality (URP)**.
    *   Watch it rebuild her skin/eyes/hair automatically.
4.  **Scene Setup:**
    *   Drag the generated Prefab into the Scene.
    *   Add `KellyAvatarController.cs` (we will write this).
    *   Connect the "Viseme" blendshapes to our Audio analyzer.

---

## 🚀 Next Action
Once you have `Kelly_Live_v1.fbx`, drop it into the Unity project folder.
Then tell me: *"Kelly is in the building."*










