# Unity Testing Guide for Complete Beginners
**You've never used Unity? No problem! Follow these exact steps.**

---

## Part 1: Installing Unity (If You Don't Have It)

### Step 1: Download Unity Hub
1. Go to: https://unity.com/download
2. Click the big blue "Download Unity Hub" button
3. Install Unity Hub (just like any other app)
4. Open Unity Hub after installation

### Step 2: Install Unity Editor
1. In Unity Hub, click "Installs" on the left sidebar
2. Click "Install Editor" (blue button, top right)
3. Choose version **2022.3 LTS** (Long Term Support - most stable)
4. Click "Next"
5. Check these boxes:
   - ✅ Android Build Support (if you have Android phone)
   - ✅ iOS Build Support (if you have iPhone)
   - ✅ Documentation (helpful for beginners)
6. Click "Next" → "Install"
7. Wait 20-30 minutes (it's a big download)
8. ☕ Take a coffee break!

---

## Part 2: Opening Our Project

### Step 3: Add the Project to Unity Hub
1. In Unity Hub, click "Projects" on the left sidebar
2. Click "Add" (top right)
3. Navigate to your project folder:
   ```
   /home/user/curiouskelly/digital-kelly/engines/kelly_unity_player
   ```
4. Click "Add Project" or "Select Folder"
5. You should now see "kelly_unity_player" in your projects list

### Step 4: Open the Project
1. Click on "kelly_unity_player" in Unity Hub
2. Unity Editor will open (takes 1-2 minutes first time)
3. You might see some import messages - **this is normal**
4. Wait for everything to finish loading (you'll see "Hold on..." at the bottom)

**What you should see:**
- A window with lots of panels
- A "Project" panel at the bottom with folders
- A "Hierarchy" panel on the left
- A big "Scene" view in the middle
- An "Inspector" panel on the right

---

## Part 3: Understanding the Unity Interface (Quick Tour)

### The Main Panels (Don't worry, we'll only use a few!)

```
┌─────────────────────────────────────────────────┐
│  Menu Bar (File, Edit, Assets, etc.)           │
├──────────┬──────────────────────┬───────────────┤
│          │                      │               │
│ Hierarchy│   Scene View         │   Inspector   │
│ (left)   │   (middle - big)     │   (right)     │
│          │                      │               │
│ List of  │   Your 3D world      │   Details of  │
│ objects  │   Visual editor      │   selected    │
│          │                      │   object      │
├──────────┴──────────────────────┴───────────────┤
│                                                  │
│  Project Panel (bottom)                          │
│  Your files and folders                          │
│                                                  │
└──────────────────────────────────────────────────┘
```

**You only need to know:**
- **Project Panel** (bottom): Where our audio files are
- **Scene View** (middle): Where Kelly avatar will appear
- **Play Button** (top center): The triangle ▶️ button

---

## Part 4: Finding and Opening Our Scene

### Step 5: Locate the Kelly Scene
1. Look at the **Project panel** (bottom of screen)
2. You'll see folders like "Assets", "Kelly", "Scripts", etc.
3. Click to expand: **Assets → Kelly → Scenes**
   - Click the little arrow (▶) next to "Assets"
   - Click the arrow next to "Kelly"
   - Click the arrow next to "Scenes"
4. You should see a file called **"KellyAvatar"** or **"KellyAvatar.unity"**

### Step 6: Open the Scene
1. **Double-click** on "KellyAvatar.unity"
2. Wait 5-10 seconds
3. The Scene view (middle panel) should now show something
   - You might see a 3D avatar
   - Or maybe just an empty space
   - Or some objects
   - **All of these are normal!**

---

## Part 5: Your First Test - Press Play!

### Step 7: Enter Play Mode
1. Look at the **very top center** of the Unity window
2. You'll see three buttons:
   - ▶️ Play (triangle)
   - ⏸️ Pause (two vertical lines)
   - ⏭️ Step (arrow with line)
3. Click the **▶️ Play** button

**What should happen:**
- The Play button turns **blue**
- The Scene view might change
- You might see numbers/text appear (like FPS counter)
- Bottom of screen might show some messages

**This is Play Mode - you're now "running" the scene!**

### Step 8: Look for the FPS Counter
While in Play Mode (Play button is blue):
1. Look at the **top-left corner** of the Scene view
2. You should see text like:
   ```
   FPS: 60.2
   Frame: 16.5ms
   ```
3. **Is the FPS number around 60?**
   - ✅ YES (55-65): Excellent! Avatar is running smoothly
   - ⚠️ NO (30-50): Still okay, but not optimal
   - ❌ NO (<30): Something might be wrong

**Write down the FPS number - we'll need it!**

### Step 9: Check the Console for Errors
1. At the bottom of Unity, find the **Console** tab
2. If you don't see it, go to: **Window → General → Console**
3. Look for messages:
   - **White/Blue messages**: Normal info (✅ good)
   - **Yellow warning ⚠️**: Usually okay, just warnings
   - **Red errors ❌**: We need to fix these

**How many red errors do you see?** _______

### Step 10: Stop Play Mode
1. Click the **▶️ Play** button again (it's blue now)
2. It will turn back to gray
3. You're now back in Edit Mode

**IMPORTANT**: Any changes you make in Play Mode are NOT saved! Always stop Play Mode before making changes.

---

## Part 6: Testing Audio Files

### Step 11: Locate the Audio Files in Unity
1. In the **Project panel** (bottom), navigate to:
   ```
   Assets → Kelly → Audio
   ```
2. You should see 10 folders:
   - the-sun
   - puppies
   - the-ocean
   - the-moon
   - water-cycle
   - molecular-biology-dna
   - creative-writing-dna
   - poetry-dna
   - dance-expression-dna
   - negotiation-skills-dna

3. Click on one folder (e.g., "the-sun")
4. You should see 54 MP3 files with names like:
   - 2-5-welcome-en.mp3
   - 6-12-mainContent-es.mp3
   - etc.

**Can you see the audio files?**
- ✅ YES: Perfect! Unity imported them correctly
- ❌ NO: We need to troubleshoot (tell me what you see instead)

### Step 12: Test Playing an Audio File
1. In the Project panel, click on ONE audio file
   - Example: `18-35-mainContent-en.mp3`
2. Look at the **Inspector panel** (right side)
3. You should see a little **audio waveform** (wavy line graphic)
4. Press the small **▶️ Play** button in the Inspector
5. You should hear Kelly's voice!

**Can you hear the audio?**
- ✅ YES: Audio system working!
- ❌ NO: Check your computer volume, or tell me what happens

---

## Part 7: Checking the Avatar Scripts

### Step 13: Find Kelly in the Scene
1. Look at the **Hierarchy panel** (left side)
2. Look for something like:
   - "Kelly" or
   - "KellyController" or
   - "KellyAvatar" or
   - An object with "Kelly" in the name

3. **Click on that object once**
4. Look at the **Inspector panel** (right side)

**What do you see in the Inspector?**
Take a screenshot or tell me what components you see, like:
- Transform
- Kelly Avatar Controller
- Blendshape Driver 60fps
- Audio Source
- etc.

### Step 14: Verify the Scripts are Connected
In the Inspector (with Kelly selected), look for these components:

**Look for "Kelly Avatar Controller":**
- ✅ Should see: "Kelly Avatar Controller (Script)"
- ✅ Should NOT say: "Script missing" or show a warning

**Look for "Blendshape Driver 60fps":**
- ✅ Should see: "Blendshape Driver 60fps (Script)"
- ✅ Should NOT say: "Script missing"

**Tell me:**
- Do you see these scripts listed?
- Are there any with "(Script)" next to them that say "missing"?

---

## Part 8: Simple Performance Test

### Step 15: Run a 1-Minute Test
1. Click **▶️ Play** button (enter Play Mode)
2. Let it run for exactly **60 seconds** (use your phone timer)
3. Watch the FPS counter (top-left of Scene view)
4. Write down:
   - **Starting FPS**: _____ (e.g., 60.2)
   - **After 30 sec**: _____ (e.g., 59.8)
   - **After 60 sec**: _____ (e.g., 60.1)
5. Click **▶️ Play** again to stop

**Did the FPS stay stable?**
- ✅ YES (within 55-65 entire time): Excellent!
- ⚠️ Dropped a bit (50-60): Okay, but we should investigate
- ❌ Dropped a lot (<50): We need to optimize

---

## Part 9: What to Report Back to Me

### Please tell me:

**1. Unity Version:**
- Which version did you install? (e.g., 2022.3.10f1)
- Go to: **Unity Editor → About Unity** to see version

**2. Project Opening:**
- ✅ Did the project open without errors?
- ❌ Were there any error messages when opening?

**3. Scene:**
- ✅ Did you find and open KellyAvatar.unity?
- What do you see in the Scene view when you open it?

**4. Play Mode:**
- ✅ Did Play Mode work (button turned blue)?
- What FPS do you see? _______

**5. Console:**
- How many errors (red messages)? _______
- Copy the first error message if any

**6. Audio Files:**
- ✅ Can you see the 10 lesson folders in Assets/Kelly/Audio?
- ✅ Can you play an audio file and hear it?

**7. Scripts:**
- ✅ Can you find Kelly in the Hierarchy?
- ✅ Do you see "Kelly Avatar Controller (Script)" in Inspector?
- ❌ Are any scripts showing as "missing"?

**8. Performance:**
- FPS during 1-minute test: Start _____, End _____
- Did it stay stable?

---

## Part 10: Common Beginner Issues (If Something Goes Wrong)

### Issue: "Unity Hub won't open the project"
**Fix:**
1. Make sure you installed Unity 2022.3 LTS
2. Try: Unity Hub → Projects → Remove project → Add it again
3. Make sure the path is correct: `digital-kelly/engines/kelly_unity_player`

### Issue: "I see lots of errors in red"
**Don't panic!** Tell me:
1. What does the first error say?
2. How many errors total?
3. Screenshot if possible

### Issue: "I can't find the Kelly scene"
**Try this:**
1. In Project panel, look at the top
2. There's a search box - type "KellyAvatar"
3. Any .unity files show up?

### Issue: "FPS is very low (like 15-20)"
**Possible causes:**
1. Your computer might not have a good GPU
2. Something else is running in background
3. Unity quality settings might be too high

**Try:**
1. Close other apps
2. Go to: **Edit → Project Settings → Quality**
3. Select "Low" or "Medium" from dropdown

### Issue: "I can't hear any audio"
**Check:**
1. Computer volume is up
2. Unity isn't muted
3. Try a different audio file
4. Check: **Edit → Project Settings → Audio** (Device should be set)

### Issue: "Play button doesn't do anything"
**Try:**
1. Save the scene: **File → Save** (Ctrl+S)
2. Look at Console for errors
3. Try: **File → New Scene** → **File → Open Scene** → Select KellyAvatar.unity

---

## Part 11: Next Steps After Basic Testing

### Once you've completed the above and reported back:

**If everything works (FPS ~60, no errors, audio plays):**
1. ✅ We'll test the avatar animation next
2. ✅ We'll test switching between age variants
3. ✅ We'll build for your phone/device
4. ✅ We'll run full performance tests

**If there are issues:**
1. Tell me exactly what you see
2. Copy any error messages
3. Take screenshots if helpful
4. I'll guide you through fixes step-by-step

---

## Quick Reference Card (Print This!)

```
╔══════════════════════════════════════════════════╗
║           UNITY QUICK REFERENCE                  ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  ▶️ PLAY BUTTON - Top center (run the scene)    ║
║  ⏸️ PAUSE - Next to Play (pause execution)      ║
║  ⏭️ STEP - Next to Pause (advance one frame)    ║
║                                                  ║
║  SAVE: Ctrl+S or Cmd+S (Mac)                    ║
║  UNDO: Ctrl+Z or Cmd+Z (Mac)                    ║
║                                                  ║
║  PANELS:                                         ║
║  - Hierarchy (left): Objects in scene           ║
║  - Inspector (right): Details of selected       ║
║  - Project (bottom): Your files                 ║
║  - Console: Window → General → Console          ║
║                                                  ║
║  CAMERA CONTROLS (Scene View):                  ║
║  - Right-click + drag: Rotate camera            ║
║  - Middle-click + drag: Pan camera              ║
║  - Scroll wheel: Zoom in/out                    ║
║  - F key: Focus on selected object              ║
║                                                  ║
║  IMPORTANT:                                      ║
║  - Changes in Play Mode are NOT saved!          ║
║  - Always stop Play Mode before editing         ║
║  - Save often with Ctrl+S                       ║
║                                                  ║
╚══════════════════════════════════════════════════╝
```

---

## You're Ready!

1. Start with **Part 1** if you don't have Unity
2. Follow each step exactly
3. Take your time - there's no rush
4. Report back your findings from **Part 9**
5. I'll be here to help with every step!

**Don't worry if something doesn't work perfectly** - that's what testing is for! Just tell me what you see, and I'll guide you through fixing it.

Good luck! You've got this! 🚀
