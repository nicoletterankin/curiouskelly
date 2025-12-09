# Unity 3D Avatar - Required C# Methods

**Status:** 🚨 BLOCKING 3D MODE  
**Priority:** HIGH  
**Impact:** 3D avatar loads but crashes when JavaScript tries to communicate

---

## Problem

When 3D mode is enabled in `learn.html`, Unity loads successfully but crashes with these errors:

```
SendMessage: object Kelly_Live_v2 does not have receiver for function SetExpression!
SendMessage: object Kelly_Live_v2 does not have receiver for function StartLipSync!
SendMessage: object Kelly_Live_v2 does not have receiver for function StopLipSync!
SendMessage: object Kelly_Live_v2 does not have receiver for function PlayAnimation!
```

---

## Root Cause

The Unity GameObject `Kelly_Live_v2` exists, but the attached C# script (`KellyAvatarController.cs`) is missing the methods that JavaScript is trying to call.

---

## Required C# Methods

### File: `KellyAvatarController.cs`

Location: `digital-kelly/engines/Kelly_Engine_V2/onlykelly/Assets/KellyAvatarController.cs`

Add these public methods to the existing script:

```csharp
using UnityEngine;
using System.Collections;

public class KellyAvatarController : MonoBehaviour
{
    [Header("Face Configuration")]
    public SkinnedMeshRenderer faceMesh; // Drag Kelly's face mesh here in Inspector
    public Animator animator; // Drag Kelly's Animator here
    
    [Header("Expression Settings")]
    public int visemeMultiplier = 100;
    
    // Current state
    private string currentExpression = "curious";
    private bool isSpeaking = false;

    // ═══════════════════════════════════════════════════════════════════
    // PUBLIC METHODS (called from JavaScript via SendMessage)
    // ═══════════════════════════════════════════════════════════════════

    /// <summary>
    /// Set Kelly's expression
    /// Called from JavaScript: unityInstance.SendMessage('Kelly_Live_v2', 'SetExpression', 'curious')
    /// </summary>
    /// <param name="expressionName">Expression: curious, explaining, wisdom, listening, celebrating</param>
    public void SetExpression(string expressionName)
    {
        Debug.Log($"[KellyAvatar] SetExpression: {expressionName}");
        currentExpression = expressionName;

        if (animator == null)
        {
            Debug.LogWarning("[KellyAvatar] Animator not assigned!");
            return;
        }

        // Trigger animation state based on expression
        switch (expressionName.ToLower())
        {
            case "curious":
                animator.SetTrigger("Curious");
                break;
            case "explaining":
                animator.SetTrigger("Explaining");
                break;
            case "wisdom":
                animator.SetTrigger("Wisdom");
                break;
            case "listening":
                animator.SetTrigger("Listening");
                break;
            case "celebrating":
                animator.SetTrigger("Celebrating");
                break;
            default:
                Debug.LogWarning($"[KellyAvatar] Unknown expression: {expressionName}");
                break;
        }
    }

    /// <summary>
    /// Start lip-sync animation
    /// Called from JavaScript: unityInstance.SendMessage('Kelly_Live_v2', 'StartLipSync', 'Hello world')
    /// </summary>
    /// <param name="textToSpeak">The text Kelly is speaking (for future viseme generation)</param>
    public void StartLipSync(string textToSpeak)
    {
        Debug.Log($"[KellyAvatar] StartLipSync: {textToSpeak}");
        isSpeaking = true;

        if (animator == null)
        {
            Debug.LogWarning("[KellyAvatar] Animator not assigned!");
            return;
        }

        // Set speaking animation parameter
        animator.SetBool("IsSpeaking", true);

        // TODO: In future, parse textToSpeak and generate viseme sequence
        // For now, just trigger generic talking animation
    }

    /// <summary>
    /// Stop lip-sync animation
    /// Called from JavaScript: unityInstance.SendMessage('Kelly_Live_v2', 'StopLipSync', '')
    /// </summary>
    public void StopLipSync()
    {
        Debug.Log("[KellyAvatar] StopLipSync");
        isSpeaking = false;

        if (animator == null)
        {
            Debug.LogWarning("[KellyAvatar] Animator not assigned!");
            return;
        }

        // Stop speaking animation
        animator.SetBool("IsSpeaking", false);
    }

    /// <summary>
    /// Play a specific animation clip
    /// Called from JavaScript: unityInstance.SendMessage('Kelly_Live_v2', 'PlayAnimation', 'wave')
    /// </summary>
    /// <param name="animationName">Name of animation clip to play</param>
    public void PlayAnimation(string animationName)
    {
        Debug.Log($"[KellyAvatar] PlayAnimation: {animationName}");

        if (animator == null)
        {
            Debug.LogWarning("[KellyAvatar] Animator not assigned!");
            return;
        }

        // Play animation by name
        animator.Play(animationName);
    }

    /// <summary>
    /// Set phase context (optional - for future use)
    /// Called from JavaScript: unityInstance.SendMessage('Kelly_Live_v2', 'SetPhase', 'welcome')
    /// </summary>
    /// <param name="phaseName">Phase: welcome, question, wisdom</param>
    public void SetPhase(string phaseName)
    {
        Debug.Log($"[KellyAvatar] SetPhase: {phaseName}");
        
        // Optional: Adjust Kelly's posture or environment based on phase
        // For example, lean forward during questions, sit back during wisdom
    }

    // ═══════════════════════════════════════════════════════════════════
    // HELPER METHODS (optional)
    // ═══════════════════════════════════════════════════════════════════

    void Start()
    {
        // Initialize
        if (animator == null)
        {
            animator = GetComponent<Animator>();
        }

        Debug.Log("[KellyAvatar] Initialized and ready for JavaScript commands");
    }

    void Update()
    {
        // Optional: Add idle animations, blinking, breathing, etc.
    }
}
```

---

## Animation Controller Setup

### Required Animator Parameters:

Add these parameters to Kelly's Animator Controller:

| Parameter Name | Type | Default | Purpose |
|---------------|------|---------|---------|
| `IsSpeaking` | Bool | false | Controls lip-sync animation |
| `Curious` | Trigger | - | Curious expression |
| `Explaining` | Trigger | - | Explaining expression |
| `Wisdom` | Trigger | - | Wisdom expression |
| `Listening` | Trigger | - | Listening expression |
| `Celebrating` | Trigger | - | Celebrating expression |

### Animation States:

Create these animation states in the Animator:

1. **Idle** (default state)
   - Subtle breathing
   - Occasional blink
   - Neutral expression

2. **Curious**
   - Lean forward slightly
   - Raised eyebrows
   - Engaged posture

3. **Explaining**
   - Hand gestures
   - Confident posture
   - Expressive face

4. **Wisdom**
   - Calm posture
   - Thoughtful expression
   - Slower movements

5. **Listening**
   - Attentive posture
   - Nodding
   - Engaged eyes

6. **Celebrating**
   - Raised arms
   - Big smile
   - Energetic movement

7. **Speaking** (blend tree)
   - Mouth movements
   - Jaw open/close
   - Lip shapes (visemes)

---

## Inspector Configuration

### GameObject: `Kelly_Live_v2`

1. **Attach Script:**
   - Add `KellyAvatarController.cs` component

2. **Assign References:**
   - `Face Mesh`: Drag Kelly's face SkinnedMeshRenderer
   - `Animator`: Drag Kelly's Animator component
   - `Viseme Multiplier`: Set to 100

3. **Verify:**
   - GameObject name MUST be exactly `Kelly_Live_v2` (case-sensitive)
   - Script must be on the root GameObject or a direct child

---

## Rebuild Steps

After adding the methods:

1. **Open Unity Project:**
   ```
   digital-kelly/engines/Kelly_Engine_V2/onlykelly/
   ```

2. **Update Script:**
   - Open `Assets/KellyAvatarController.cs`
   - Add the methods above
   - Save

3. **Configure Animator:**
   - Open Kelly's Animator Controller
   - Add required parameters
   - Create animation states
   - Set up transitions

4. **Assign References:**
   - Select `Kelly_Live_v2` GameObject
   - Assign Face Mesh and Animator in Inspector

5. **Build WebGL:**
   - File → Build Settings → WebGL
   - Build to: `Kelly_Web_Build/`
   - Wait for build to complete

6. **Copy Build Files:**
   ```bash
   xcopy "Kelly_Web_Build\Build\*" "C:\Users\user\UI-TARS-desktop\public\unity\kelly\Build\" /E /I /Y
   xcopy "Kelly_Web_Build\StreamingAssets" "C:\Users\user\UI-TARS-desktop\public\unity\kelly\Build\StreamingAssets\" /E /I /Y
   ```

7. **Test:**
   - Open `http://localhost:8080/learn.html?day=333`
   - Click Mode button to switch to 3D
   - Verify no console errors
   - Verify Kelly's expression changes during lesson phases

---

## JavaScript Integration (Already Implemented)

The JavaScript side is already set up in:

- `public/js/unity-kelly-loader.js` (lines 200-250)
- `public/js/kelly-avatar-controller.js` (lines 150-200)
- `public/learn.html` (lines 1600-1650)

Example JavaScript calls:

```javascript
// Set expression
unityInstance.SendMessage('Kelly_Live_v2', 'SetExpression', 'curious');

// Start speaking
unityInstance.SendMessage('Kelly_Live_v2', 'StartLipSync', 'Welcome to today\'s lesson!');

// Stop speaking
unityInstance.SendMessage('Kelly_Live_v2', 'StopLipSync', '');

// Play animation
unityInstance.SendMessage('Kelly_Live_v2', 'PlayAnimation', 'wave');
```

---

## Testing Checklist

After rebuild:

- [ ] Unity build completes without errors
- [ ] WebGL files copied to `public/unity/kelly/Build/`
- [ ] `learn.html` loads without console errors
- [ ] 3D mode button works (toggle 2D ↔ 3D)
- [ ] Kelly's expression changes during lesson phases:
  - [ ] Welcome → Curious
  - [ ] Question → Explaining
  - [ ] Student selects choice → Listening
  - [ ] Kelly responds → Explaining
  - [ ] Wisdom → Wisdom
- [ ] No `SendMessage: object Kelly_Live_v2 does not have receiver` errors
- [ ] Lip-sync animation plays when Kelly speaks
- [ ] Smooth transitions between expressions

---

## Future Enhancements

Once basic methods work:

1. **Viseme Generation:**
   - Parse text and generate phoneme-based mouth shapes
   - Use Audio2Face or similar for realistic lip-sync

2. **Emotion Blending:**
   - Blend between expressions smoothly
   - Add micro-expressions (subtle eyebrow raises, smiles)

3. **Contextual Animations:**
   - Different gestures based on lesson topic
   - React to student's choice (positive/negative feedback)

4. **Environment Integration:**
   - Change background based on lesson topic
   - Add props (books, globe, etc.)

---

## Support

If Unity build fails or methods don't work:

1. **Check Console:**
   - Unity Editor Console (for build errors)
   - Browser Console (for JavaScript errors)

2. **Verify GameObject Name:**
   - MUST be exactly `Kelly_Live_v2`
   - Case-sensitive

3. **Check Script Attachment:**
   - `KellyAvatarController.cs` must be attached to `Kelly_Live_v2`

4. **Test in Unity Editor First:**
   - Use Unity's Play mode
   - Call methods manually via Inspector
   - Verify animations work before building

---

**Last Updated:** 2025-11-28  
**Status:** Awaiting Unity rebuild with new methods











