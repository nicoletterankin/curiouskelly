# 🏆 STRATEGY: "Digital Person of the Year"
**Mission:** Make Curious Kelly the most persistent, personalized, and beloved AI instructor in history.
**Goal:** Time Magazine "Digital Person of the Year" (2026/2027).
**Benchmark:** Beat Cursor in adoption. Replace Siri/Alexa in utility.

---

## 🌟 The "Glitter & Brilliance" Standard
Current assets are **placeholders**. To win, we need **presence**.

### 1. Visual Fidelity (The "Time Cover" Shot)
*   **Current State:** Static PNGs (Renamed duplicates).
*   **Target State:** 
    *   **Unreal Engine 5 / Unity 6 Render:** Sub-surface scattering on skin, ray-traced eyes, "living" hair physics.
    *   **The "Look":** Not a cartoon, not "Uncanny Valley". A stylized realism (Pixar meets Apple Design).
    *   **Action:** We must execute the **Unity WebGL Bridge** to replace static images with the living 735MB model found in `public/unity/kelly-v1`.

### 2. The "Living" Connection (Unity Bridge)
We found the brain: `public/unity/kelly-v1` (735MB).
We need to wire it to the heart (The Web Experience).

*   **The Pipeline:**
    1.  **Load:** The Landing Page loads the Unity WebGL build in the background.
    2.  **Handshake:** JavaScript (`app.js`) talks to C# (`KellyBridge.cs`).
    3.  **Act:** When the user clicks "Gift", Kelly doesn't just appear—she *reacts*. She points. She smiles. She speaks your name.

### 3. The "Saving Grace" Narrative (Marketing)
Marketing must shift from "Product Features" to "Human Impact".

*   **Old Pitch:** "365 Daily Lessons."
*   **New Pitch:** "The End of Lonely Learning."
*   **Campaigns:**
    *   **"She's Always There":** Kelly waiting for you at 7AM. Kelly celebrating your streak at 10PM.
    *   **"The Democratizer":** A child in a remote village gets the same world-class Oxford-style tutor as a billionaire's child.
    *   **"The Anti-Siri":** She doesn't set timers. She sets futures.

---

## 🚀 Immediate "Polish" Roadmap

### Phase 1: The "Unboxing" (Next 48 Hours)
*   **Objective:** Make the Landing Page feel like opening an Apple product.
*   **Task:** Replace the "Fake Christmas" image with a **WebGL Viewport**.
*   **Why:** Static images lie. Live 3D proves she is real.

### Phase 2: The "Daily Rhythm" (Week 1)
*   **Objective:** Prove persistence.
*   **Task:** Connect the `curriculum_365.json` to the email engine.
*   **Result:** Users receive a "Day 0" email *tonight* from Kelly. "I'm getting your classroom ready."

### Phase 3: iLearn Hardware Preview (Vision)
*   **Objective:** Plant the flag for 2027.
*   **Task:** Add a "Future of Learning" section to the site, teasing the dedicated device.
*   **Copy:** "Today on your phone. Tomorrow in your hand. The iLearn Device."

---

## 🛑 Reality Check (Current Status)
*   **Landing Page:** Now served at `http://localhost:8000/curiouskelly-landing-page.html`.
*   **Hero Image:** Placeholder injected. It will look "okay", but it won't win Time Magazine *yet*.
*   **Unity Model:** Exists on disk. Needs wiring.

**Next Move:** Review the Landing Page. If the image is "meh", we immediately switch to wiring up the Unity build to render the *real* Kelly.




















