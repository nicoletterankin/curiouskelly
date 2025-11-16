# Unity Testing Handoff to Cursor AI
**Handoff Date**: November 16, 2025
**Context**: Avatar 60fps integration testing for Curious Kelly project
**Your Role**: Unity testing and validation ONLY
**Content Owner**: Claude (maintains all backend, audio, content work)

---

## 🎯 Your Mission (Cursor)

You are taking over **Unity testing and validation only**. You will:
- ✅ Guide the user through Unity Editor testing
- ✅ Help them test the Kelly avatar at 60fps
- ✅ Validate audio playback in Unity
- ✅ Measure performance metrics
- ✅ Report findings back to the user (who will share with Claude)

You will **NOT**:
- ❌ Modify any backend code
- ❌ Change audio generation scripts
- ❌ Alter content or lesson files
- ❌ Touch anything outside Unity Editor

---

## 📋 Current Project State (What Claude Completed)

### ✅ COMPLETE - Do Not Touch
1. **Backend (Node.js/Express)**: Fully operational
   - Location: `curious-kellly/backend/`
   - All APIs working (lessons, voice, safety)
   - ElevenLabs voice synthesis integrated
   - Do NOT modify this

2. **Audio Generation**: 100% complete
   - 558 MP3 files generated (191.3 MB)
   - Location: `curious-kellly/backend/config/audio/`
   - Also copied to Unity: `digital-kelly/engines/kelly_unity_player/Assets/Kelly/Audio/`
   - All 10 lessons × 6 age variants × 3 languages
   - Do NOT regenerate or modify audio

3. **Content**: Complete
   - 10 lessons with 180 multilingual variants
   - Location: `curious-kellly/backend/config/lessons/`
   - Do NOT modify lesson content

4. **Unity Scripts**: All present and validated
   - Location: `digital-kelly/engines/kelly_unity_player/Assets/Kelly/Scripts/`
   - 20 C# scripts (109 KB total)
   - Do NOT modify existing scripts

### ⏳ YOUR TASK - Unity Testing
1. **Unity Scene Setup**: Create/configure test scene
2. **Performance Testing**: Measure 60fps performance
3. **Audio Integration**: Test 558 audio files in Unity
4. **Avatar Testing**: Validate avatar animation/blendshapes
5. **Report Results**: Document findings for user to share with Claude

---

## 🎬 Unity Project Overview

### Project Location
```
/home/user/curiouskelly/digital-kelly/engines/kelly_unity_player/
```

### Key Assets
```
Assets/
├── Kelly/
│   ├── Scripts/          (20 C# files - avatar system)
│   │   ├── BlendshapeDriver60fps.cs
│   │   ├── KellyAvatarController.cs
│   │   ├── GazeController.cs
│   │   ├── VisemeMapper.cs
│   │   └── ... (16 more)
│   ├── Models/           (Kelly 3D avatar FBX)
│   └── Audio/            (558 MP3 files, 192 MB)
│       ├── the-sun/      (54 files)
│       ├── puppies/      (54 files)
│       ├── the-ocean/    (54 files)
│       └── ... (7 more lessons)
```

### Unity Version
- **Required**: Unity 2022.3 LTS (Long Term Support)
- **Acceptable**: Any 2022.3.x version

---

## 📝 Step-by-Step Testing Instructions

### Phase 1: Unity Setup & Scene Creation

**Current Situation**:
- User has Unity open
- Currently viewing "kelly_placeholder web" (wrong scene)
- NO scene file exists yet for testing
- User is a complete Unity beginner

**Your First Task**:
1. Help user create a new test scene:
   ```
   File → New Scene → Basic (Built-in) → Create
   File → Save As → Assets/Kelly/KellyTest.unity
   ```

2. Verify scene saved correctly:
   - Check Hierarchy panel has: Main Camera, Directional Light
   - Project panel shows: Assets/Kelly/KellyTest.unity

### Phase 2: Audio Validation in Unity

**Test audio import**:
1. Navigate to: `Assets/Kelly/Audio/` in Project panel
2. Verify 10 lesson folders visible
3. Click on one lesson folder (e.g., "the-sun")
4. Verify 54 MP3 files visible
5. Select one audio file → Inspector panel → Press Play button
6. User should hear Kelly's voice

**Expected Result**:
- ✅ All 558 files visible and organized
- ✅ Audio plays in Unity
- ✅ No import errors

**If Issues**:
- Check Console for errors
- Reimport audio: Right-click Audio folder → Reimport

### Phase 3: Performance Baseline Test

**Simple FPS test** (no avatar yet):
1. With empty scene open, press Play button (▶️)
2. Check top-left corner for FPS counter
3. Let run for 60 seconds
4. Record:
   - Starting FPS: _____
   - Average FPS: _____
   - Minimum FPS: _____

**Expected Result**:
- ✅ Should see 60+ FPS easily (empty scene)
- ✅ No red errors in Console

### Phase 4: Kelly Avatar Setup (Advanced)

**NOTE**: This requires understanding of Unity GameObjects and Components.

**If Kelly avatar FBX exists**:
1. Find Kelly model: `Assets/Kelly/Models/`
2. Drag Kelly FBX into Hierarchy
3. Select Kelly in Hierarchy
4. In Inspector, add components:
   - Add Component → Search "Kelly Avatar Controller"
   - Add Component → Search "Blendshape Driver 60fps"
   - Add Component → "Audio Source"

**If NO Kelly model**:
- Create placeholder: GameObject → 3D Object → Capsule
- Rename to "Kelly_Placeholder"
- Still test audio system

### Phase 5: Performance Testing (With Content)

**Test with audio playback**:
1. Select Kelly (or placeholder) in Hierarchy
2. Find Audio Source component in Inspector
3. Set AudioClip: Drag an MP3 from Project panel
4. Check "Play On Awake"
5. Press Play (▶️)
6. Monitor FPS while audio plays

**Measure**:
- FPS during playback: _____
- CPU usage (if visible): _____
- Any frame drops? Yes/No
- Audio plays smoothly? Yes/No

### Phase 6: Script Validation

**Check all scripts compile**:
1. Window → Console
2. Look for compilation errors
3. If scripts have errors, they'll show in red

**Expected Result**:
- ✅ All scripts compile without errors
- ✅ Can find scripts in Add Component menu

**Common Issue**:
- If "Missing Script" errors: Scripts might need namespace fixes
- Report exact error messages

---

## 📊 Performance Targets

### Minimum Acceptable Performance
```
FPS:          55-65 fps (stable)
Frame Time:   <18ms
Console:      0 red errors
Audio:        Plays smoothly, no crackling
```

### Ideal Performance
```
FPS:          60+ fps (locked)
Frame Time:   <16.67ms
Console:      0 errors, minimal warnings
Audio:        Perfect playback
Memory:       <500MB
```

### Red Flags (Report These)
```
FPS:          <50 fps
Frame Time:   >20ms
Console:      Multiple red errors
Audio:        Doesn't play or crackles
Crashes:      Any Unity crashes
```

---

## 🐛 Common Unity Issues & Fixes

### Issue: "Can't find scene"
**Reason**: No scene file exists yet
**Fix**: Create new scene (Phase 1)

### Issue: "Audio files not visible"
**Reason**: Unity hasn't imported them yet
**Fix**:
1. Right-click Assets folder → Reimport All
2. Wait for import to complete
3. Check Assets/Kelly/Audio/

### Issue: "Scripts show as missing"
**Reason**: Namespace or compilation error
**Fix**:
1. Check Console for errors
2. Double-click error to see script
3. Report exact error message

### Issue: "FPS very low (<30)"
**Reason**: Quality settings too high or hardware limitation
**Fix**:
1. Edit → Project Settings → Quality
2. Select "Low" or "Medium"
3. Test again

### Issue: "Nothing happens when I press Play"
**Reason**: Scene might not be saved
**Fix**:
1. File → Save (Ctrl+S)
2. Try Play again

---

## 📋 Testing Checklist (Complete This)

### Basic Setup
- [ ] Unity 2022.3 LTS installed
- [ ] Project opens without errors
- [ ] Can navigate Project panel
- [ ] Console has no red errors on load

### Scene Creation
- [ ] Created KellyTest.unity scene
- [ ] Scene saved in Assets/Kelly/
- [ ] Can press Play and see scene run
- [ ] FPS counter visible

### Audio Validation
- [ ] Can see Assets/Kelly/Audio/ folder
- [ ] Can see 10 lesson subfolders
- [ ] Selected one MP3 file
- [ ] Pressed Play in Inspector
- [ ] Heard Kelly's voice
- [ ] All 558 files present (count verified)

### Performance Baseline
- [ ] Empty scene FPS: _____ fps
- [ ] Stable over 60 seconds: Yes/No
- [ ] No frame drops: Yes/No
- [ ] Console clear: Yes/No

### Audio Playback Test
- [ ] Added Audio Source to GameObject
- [ ] Assigned MP3 file to AudioClip
- [ ] Pressed Play in scene
- [ ] Audio plays: Yes/No
- [ ] FPS during audio: _____ fps
- [ ] Audio quality good: Yes/No

### Script Validation
- [ ] All 20 scripts visible in Project
- [ ] No compilation errors: Yes/No
- [ ] Can find scripts in Add Component: Yes/No
- [ ] Any "Missing Script" errors: Yes/No

---

## 📝 Report Template (Give This to User)

```markdown
## Unity Testing Report

**Date**: [Today's date]
**Tester**: Cursor AI
**Unity Version**: [Version from Unity Hub]

### Environment
- OS: [Windows/Mac/Linux]
- GPU: [If known]
- Unity Version: [e.g., 2022.3.10f1]

### Test Results

#### Scene Setup
- Scene created: ✅/❌
- Scene location: Assets/Kelly/KellyTest.unity
- Issues: [None or describe]

#### Audio Import
- Files visible: ✅/❌ ([Count] files found)
- Audio plays: ✅/❌
- Quality: ✅/❌
- Issues: [None or describe]

#### Performance
- Empty scene FPS: ___ fps
- With audio FPS: ___ fps
- Stable: ✅/❌
- Frame drops: ✅/❌

#### Scripts
- Compilation: ✅/❌
- Errors: [Number of errors]
- Error messages: [Copy/paste if any]

#### Console Errors
[Copy/paste any red errors, or write "None"]

### Recommendations
[What works, what needs fixing]

### Next Steps
[What should be done next]
```

---

## 🚫 Out of Scope (Do Not Do)

**DO NOT modify**:
- ❌ Backend code (`curious-kellly/backend/`)
- ❌ Audio files (`.mp3` files)
- ❌ Lesson content (`.json` files)
- ❌ Audio generation scripts (`.py` files)
- ❌ Environment variables (`.env` files)
- ❌ Git repository (commits, pushes)

**DO NOT create**:
- ❌ New audio files
- ❌ New lesson content
- ❌ New backend endpoints
- ❌ New voice models

**Your job is ONLY Unity testing** - measuring FPS, validating audio, checking scripts, and reporting results.

---

## 🤝 Handoff Protocol

### When Testing is Complete

**User will report back to Claude with**:
1. Your completed test report (use template above)
2. Any issues found
3. Performance metrics
4. Screenshots if needed

**Claude will then**:
1. Review your findings
2. Make any necessary fixes to backend/content
3. Provide next steps for integration
4. Handle all non-Unity work

### Division of Responsibilities

**Cursor (You)**: Unity Editor only
- Scene setup
- Audio testing in Unity
- Performance measurement
- Script validation
- Unity-specific troubleshooting

**Claude (Content Owner)**: Everything else
- Backend development
- Audio generation
- Lesson content
- Voice synthesis
- Integration architecture
- Production deployment

---

## 📚 Helpful Unity Tips for Beginner User

### Key Shortcuts
- `Ctrl+S` or `Cmd+S`: Save
- `Ctrl+Z` or `Cmd+Z`: Undo
- `F`: Focus on selected object
- `Ctrl+D`: Duplicate object

### Important Panels
- **Hierarchy** (left): Objects in scene
- **Inspector** (right): Properties of selected object
- **Project** (bottom): All files
- **Console**: Window → General → Console
- **Scene** (middle): 3D view

### Camera Controls in Scene View
- Right-click + drag: Rotate camera
- Middle-click + drag: Pan camera
- Scroll wheel: Zoom in/out
- Alt+click + drag: Orbit around object

### Play Mode Warning
**IMPORTANT**: Changes made in Play Mode are NOT saved!
- Always stop Play Mode before making permanent changes
- Play button turns blue when active
- Click it again to stop

---

## 🎯 Success Criteria

### Your testing is successful if:
1. ✅ User can open Unity and navigate
2. ✅ Scene created and saved
3. ✅ All 558 audio files visible and playable
4. ✅ FPS measured and documented
5. ✅ Console errors (if any) documented
6. ✅ Report completed and given to user
7. ✅ User understands next steps

### Your testing is complete when:
- User can share the report with Claude
- All performance metrics documented
- Any blockers identified
- User knows how to proceed

---

## 🚀 Start Here

**First thing to do**:
1. Confirm user has Unity open
2. Ask: "What do you see in the Scene view right now?"
3. Guide them to create the KellyTest scene
4. Start working through the checklist

**Remember**:
- User is a complete Unity beginner
- Explain every step clearly
- Use exact button names and locations
- Be patient and encouraging
- Focus ONLY on Unity testing

**Ready? Start with Phase 1 and guide them step by step!**

---

## 📞 Questions for Cursor

If you're unclear on anything:
- **Unity setup**: Proceed with standard 2022.3 LTS setup
- **Missing assets**: Work with what's available, report what's missing
- **Script errors**: Document and report, don't try to fix C# code
- **Performance issues**: Measure and report, let Claude handle fixes

**Your goal**: Get clean performance metrics and validation, then hand back to Claude for any fixes needed.

---

**End of Handoff Document**

Cursor: You now have full context. Begin testing and guide the user through Unity validation!
