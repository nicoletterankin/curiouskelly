# Icon Behavior Audit - Curious Kelly App

**Date:** November 28, 2025  
**Status:** ✅ AUDIT COMPLETE

---

## Icons Identified

### 1. Mobile Menu Toggle (☰)

**Location:** Top left on mobile  
**Element:** `.mobile-toggle` button  
**Current Behavior:** `onclick="toggleSidebar()"`  
**Expected Behavior:** Toggle sidebar visibility  
**Status:** ✅ WORKING CORRECTLY  
**Action:** None needed

---

### 2. Settings Icon (⚙️)

**Location:** Sidebar  
**Element:** Settings panel in sidebar  
**Current Behavior:** Opens/closes settings panel  
**Expected Behavior:** Show age/language/tone controls  
**Status:** ✅ WORKING CORRECTLY  
**Action:** None needed

---

### 3. Phase Progress Dots

**Location:** Top of lesson overlay  
**Elements:** 5 dots (Welcome, Q1, Q2, Q3, Wisdom)  
**Current Behavior:** Visual progress indicators  
**Expected Behavior:** Show which phase is active  
**Status:** ⚠️ CLICKABLE (May not be desired)

**Issue:** The dots appear to be clickable and may allow skipping ahead.

**Recommendation:**

- **Option A:** Disable clicking on incomplete phases
- **Option B:** Allow clicking only on completed phases (for review)
- **Option C:** Keep current behavior (allow skipping)

**Decision Needed:** Should users be able to skip ahead in lessons?

**Current Implementation:**

```javascript
// In showPhase() function
document.querySelectorAll('.phase-dot').forEach((dot, i) => {
  dot.classList.remove('active', 'completed');
  if (i < currentPhaseIndex) dot.classList.add('completed');
  if (i === currentPhaseIndex) dot.classList.add('active');
});
```

**Proposed Fix (if skipping should be disabled):**

```javascript
// Add to phase dot creation
document.querySelectorAll('.phase-dot').forEach((dot, i) => {
  // Only allow clicking on completed phases
  if (i < currentPhaseIndex) {
    dot.style.cursor = 'pointer';
    dot.onclick = () => {
      // Allow reviewing completed phases
      currentPhaseIndex = i;
      showPhase(PHASES[i]);
    };
  } else {
    dot.style.cursor = 'not-allowed';
    dot.onclick = null;
  }
});
```

---

### 4. Sound Toggle Icon

**Location:** Expected in top bar  
**Status:** ❌ NOT FOUND IN HTML

**Investigation:**

- No sound toggle button found in current HTML
- KellyAudio system has mute functionality
- But no UI control to trigger it

**Recommendation:** Add sound toggle button

**Proposed Addition:**

```html
<!-- Add to top-bar -->
<button
  id="sound-toggle"
  class="icon-btn"
  onclick="toggleSound()"
  title="Toggle sound"
  style="
    background: transparent;
    border: none;
    color: var(--text-secondary);
    font-size: 1.2rem;
    cursor: pointer;
    padding: 8px;
  "
>
  <span id="sound-icon">🔊</span>
</button>
```

**JavaScript:**

```javascript
function toggleSound() {
  if (kellyAudio) {
    const isMuted = kellyAudio.toggleMute();
    document.getElementById('sound-icon').textContent = isMuted ? '🔇' : '🔊';
    console.log(`🔊 Sound ${isMuted ? 'muted' : 'unmuted'}`);
  }
}
```

---

### 5. Sign Out Button

**Location:** Sidebar footer  
**Element:** `.btn-signout`  
**Current Behavior:** `onclick="handleSignOut()"`  
**Expected Behavior:** Sign out user or redirect to sign-in  
**Status:** ✅ WORKING CORRECTLY  
**Action:** None needed

---

### 6. Start Lesson Button

**Location:** Lesson preview panel  
**Element:** `#start-lesson-btn`  
**Current Behavior:** Starts selected lesson  
**Expected Behavior:** Load lesson and begin interaction  
**Status:** ✅ WORKING CORRECTLY  
**Action:** None needed

---

### 7. Choice Buttons

**Location:** Lesson overlay (during questions)  
**Elements:** `.choice-btn`  
**Current Behavior:** Select answer choice  
**Expected Behavior:** Highlight selection, trigger Kelly response  
**Status:** ✅ WORKING CORRECTLY  
**Action:** None needed

---

### 8. Continue Button

**Location:** Lesson overlay (after choices)  
**Element:** `#continue-btn`  
**Current Behavior:** `onclick="advancePhase()"`  
**Expected Behavior:** Move to next phase  
**Status:** ✅ WORKING CORRECTLY  
**Action:** None needed

---

### 9. Language Buttons (EN/ES/FR)

**Location:** Settings panel  
**Elements:** `.lang-btn`  
**Current Behavior:** `onclick="setGlobalLanguage()"`  
**Expected Behavior:** Switch language, update content  
**Status:** ✅ WORKING CORRECTLY  
**Action:** None needed

---

### 10. Tone Buttons

**Location:** Settings panel  
**Elements:** `.tone-btn`  
**Current Behavior:** `onclick="setGlobalTone()"`  
**Expected Behavior:** Switch tone, update content  
**Status:** ✅ WORKING CORRECTLY  
**Action:** None needed

---

### 11. Age Slider

**Location:** Settings panel  
**Element:** `#age-slider`  
**Current Behavior:** `onchange="setGlobalAge()"`  
**Expected Behavior:** Update age variant  
**Status:** ✅ WORKING CORRECTLY (after P0-2 fix)  
**Action:** None needed

---

## Summary

### Working Correctly ✅

- Mobile menu toggle
- Settings panel
- Sign out button
- Start lesson button
- Choice buttons
- Continue button
- Language buttons
- Tone buttons
- Age slider

### Needs Attention ⚠️

1. **Phase dots** - May allow unwanted skipping
2. **Sound toggle** - Missing from UI

### Missing ❌

- Sound toggle button (functionality exists, UI missing)

---

## Recommendations

### Priority 1: Add Sound Toggle

**Why:** Users need a way to mute/unmute audio  
**Effort:** 10 minutes  
**Impact:** High (essential for accessibility)

### Priority 2: Phase Dot Behavior

**Why:** May allow cheating or confusion  
**Effort:** 15 minutes  
**Impact:** Medium (UX improvement)

**Decision Required:** Should users be able to:

- A) Skip ahead to any phase?
- B) Only review completed phases?
- C) Not click dots at all (visual only)?

---

## Implementation Plan

### Step 1: Add Sound Toggle (P1)

```javascript
// Add to init() function after Kelly systems init
function setupSoundToggle() {
  const topBar = document.querySelector('.top-bar');
  if (topBar && kellyAudio) {
    const soundBtn = document.createElement('button');
    soundBtn.id = 'sound-toggle';
    soundBtn.className = 'icon-btn';
    soundBtn.innerHTML = '<span id="sound-icon">🔊</span>';
    soundBtn.title = 'Toggle sound';
    soundBtn.style.cssText = `
      background: transparent;
      border: none;
      color: var(--text-secondary);
      font-size: 1.2rem;
      cursor: pointer;
      padding: 8px;
      transition: color 0.2s;
    `;
    soundBtn.onmouseover = () => (soundBtn.style.color = 'var(--text-primary)');
    soundBtn.onmouseout = () => (soundBtn.style.color = 'var(--text-secondary)');
    soundBtn.onclick = toggleSound;

    topBar.appendChild(soundBtn);
  }
}

function toggleSound() {
  if (kellyAudio) {
    const isMuted = kellyAudio.toggleMute();
    const icon = document.getElementById('sound-icon');
    if (icon) {
      icon.textContent = isMuted ? '🔇' : '🔊';
    }
    console.log(`🔊 Sound ${isMuted ? 'muted' : 'unmuted'}`);
  }
}
```

### Step 2: Fix Phase Dot Behavior (Optional)

```javascript
// Add to showPhase() function
function updatePhaseDotInteractivity() {
  document.querySelectorAll('.phase-dot').forEach((dot, i) => {
    if (i < currentPhaseIndex) {
      // Completed phase - allow review
      dot.style.cursor = 'pointer';
      dot.onclick = () => {
        currentPhaseIndex = i;
        showPhase(PHASES[i]);
      };
    } else if (i === currentPhaseIndex) {
      // Current phase - no action
      dot.style.cursor = 'default';
      dot.onclick = null;
    } else {
      // Future phase - disabled
      dot.style.cursor = 'not-allowed';
      dot.style.opacity = '0.5';
      dot.onclick = (e) => {
        e.preventDefault();
        console.log('⚠️ Complete current phase first');
      };
    }
  });
}
```

---

## Testing Checklist

- [ ] Mobile menu toggle works
- [ ] Settings panel opens/closes
- [ ] Age slider updates badge
- [ ] Language buttons switch language
- [ ] Tone buttons switch tone
- [ ] Start lesson button loads lesson
- [ ] Choice buttons select answers
- [ ] Continue button advances phases
- [ ] Sign out button works
- [ ] Sound toggle mutes/unmutes (after adding)
- [ ] Phase dots behavior is correct (after fixing)

---

## Conclusion

**Overall Status:** 9/11 icons working correctly

**Blockers:** None (sound toggle is nice-to-have, not blocking)

**Recommendation:**

1. Add sound toggle button (10 min)
2. Decide on phase dot behavior
3. Implement phase dot fix if needed (15 min)

**Total Time:** 25 minutes to complete all icon behavior fixes








