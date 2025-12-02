# 📱 SIDEBAR REFERENCE - learn.html

## Current Sidebar Structure (Lines 712-743)

### HTML Structure

```html
<div class="action-buttons">
  <!-- Age Button -->
  <button class="action-btn" id="btn-age" aria-label="Age variant">
    <div class="icon-wrap">
      🎂
      <span class="badge" id="badge-age">18</span>
    </div>
    <span class="label">Age</span>
  </button>

  <!-- Language Button -->
  <button class="action-btn" id="btn-language" aria-label="Language">
    <div class="icon-wrap">
      🌍
      <span class="badge" id="badge-language">EN</span>
    </div>
    <span class="label">Lang</span>
  </button>

  <!-- Difficulty Button -->
  <button class="action-btn" id="btn-difficulty" aria-label="Difficulty">
    <div class="icon-wrap">
      🎯
      <span class="badge" id="badge-difficulty">2</span>
    </div>
    <span class="label">Level</span>
  </button>

  <!-- Share Button -->
  <button class="action-btn" id="btn-share" aria-label="Share">
    <div class="icon-wrap">↗️</div>
    <span class="label">Share</span>
  </button>

  <!-- Sound Button (different style) -->
  <button class="sound-btn" id="btn-sound" aria-label="Sound">🔊</button>
</div>
```

---

## CSS Classes

### `.action-buttons` (Container)

```css
.action-buttons {
  position: absolute;
  right: 12px;
  bottom: calc(180px + var(--safe-bottom));
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--action-gap); /* 20px */
  z-index: 100;
}
```

### `.action-btn` (Button)

```css
.action-btn {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  background: none;
  border: none;
  cursor: pointer;
  -webkit-tap-highlight-color: transparent;
}
```

### `.icon-wrap` (Icon Container)

```css
.action-btn .icon-wrap {
  width: var(--action-btn-size); /* 48px */
  height: var(--action-btn-size); /* 48px */
  border-radius: 50%;
  background: var(--tiktok-glass); /* rgba(0, 0, 0, 0.5) */
  backdrop-filter: blur(8px);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: var(--action-icon-size); /* 28px */
  transition: all 0.2s ease;
  position: relative; /* For badge positioning */
}
```

### `.badge` (Number Badge)

```css
.action-btn .badge {
  position: absolute;
  top: -4px;
  right: -4px;
  min-width: 18px;
  height: 18px;
  background: var(--tiktok-accent); /* #fe2c55 */
  border-radius: 9px;
  font-size: 10px;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 4px;
}
```

### `.label` (Text Label)

```css
.action-btn .label {
  font-size: 10px;
  color: var(--tiktok-text); /* #fff */
  font-weight: 500;
}
```

### `.sound-btn` (Special Sound Button)

```css
.sound-btn {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: var(--tiktok-glass);
  backdrop-filter: blur(8px);
  border: 2px solid var(--tiktok-text-muted);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  animation: spin 3s linear infinite;
  animation-play-state: running;
}
```

---

## Button Pattern

### Standard Button with Badge

```html
<button class="action-btn" id="btn-{name}">
  <div class="icon-wrap">
    {emoji}
    <span class="badge" id="badge-{name}">{value}</span>
  </div>
  <span class="label">{Label}</span>
</button>
```

### Button without Badge

```html
<button class="action-btn" id="btn-{name}">
  <div class="icon-wrap">{emoji}</div>
  <span class="label">{Label}</span>
</button>
```

---

## Modal Pattern

### Modal HTML (Lines 795-845 for Age example)

```html
<div class="modal-overlay" id="modal-{name}">
  <div class="modal-content">
    <div class="modal-header">
      <span class="modal-title">{emoji} {Title}</span>
      <button class="modal-close" data-close="{name}">×</button>
    </div>

    <div class="variant-option selected" data-{name}="{value}">
      <div class="variant-radio"></div>
      <div class="variant-info">
        <div class="variant-name">{Option Name}</div>
        <div class="variant-desc">{Description}</div>
      </div>
    </div>

    <!-- More options... -->
  </div>
</div>
```

### Modal JavaScript (Lines 1435-1460)

```javascript
// Open modal
function openModal(type) {
  document.getElementById(`modal-${type}`).classList.add('open');
}

// Close modal
function closeModal(type) {
  document.getElementById(`modal-${type}`).classList.remove('open');
}

// Select variant
function selectVariant(type, value) {
  state.variants[type] = value;
  savePreferences();
  updateUI();

  // Re-render current phase with new variant
  if (state.lesson) {
    renderPhase(state.lesson.phases[state.currentPhase - 1]);
  }

  closeModal(type);
  showToast(`${type.charAt(0).toUpperCase() + type.slice(1)} updated!`);
}
```

### Button Handler (Lines 1487-1489)

```javascript
document.getElementById('btn-{name}').onclick = () => openModal('{name}');
```

### Modal Handlers (Lines 1509-1526)

```javascript
// Close button
document.querySelectorAll('.modal-close').forEach((btn) => {
  btn.onclick = () => closeModal(btn.dataset.close);
});

// Click outside to close
document.querySelectorAll('.modal-overlay').forEach((modal) => {
  modal.onclick = (e) => {
    if (e.target === modal) modal.classList.remove('open');
  };
});

// Option selection
document.querySelectorAll('[data-{name}]').forEach((el) => {
  el.onclick = () => selectVariant('{name}', el.dataset.{name});
});
```

---

## Visual Layout

```
┌─────────────────────────────────────┐
│                                     │
│                                     │
│                                     │
│                                     │
│                 Kelly               │
│                Avatar               │
│                                     │
│                                     │
│                                 ┌─┐ │
│                                 │🎂│ ← Age (badge: 18)
│                                 └─┘ │
│                                 ┌─┐ │
│                                 │🌍│ ← Language (badge: EN)
│                                 └─┘ │
│                                 ┌─┐ │
│                                 │🎯│ ← Difficulty (badge: 2)
│                                 └─┘ │
│                                 ┌─┐ │
│                                 │↗️│ ← Share
│                                 └─┘ │
│                                 ┌─┐ │
│                                 │🔊│ ← Sound (spinning)
│                                 └─┘ │
│                                     │
│  [Topic & Speech Bubble]            │
│  [Choices]                          │
│                                     │
└─────────────────────────────────────┘
```

---

## Suggested Icon Options

### For Tone Button

- 🎭 Theater masks (tone/mood)
- 🎨 Artist palette (style)
- 💭 Thought bubble (thinking style)
- 🌟 Sparkle (vibe)
- **Recommended:** 🎭 with badge showing C/P/S

### For 2D/3D Toggle

- 🎬 Movie camera (mode)
- 📐 Cube (3D)
- 🖼️ Frame (2D)
- 🔄 Switch icon
- **Recommended:** 📐 for 3D, 🖼️ for 2D (toggle between them)

---

## Adding New Buttons - Checklist

### 1. HTML (in `.action-buttons`)

```html
<button class="action-btn" id="btn-tone" aria-label="Tone">
  <div class="icon-wrap">
    🎭
    <span class="badge" id="badge-tone">C</span>
  </div>
  <span class="label">Tone</span>
</button>
```

### 2. Modal HTML (before `<!-- Toast Container -->`)

```html
<div class="modal-overlay" id="modal-tone">
  <div class="modal-content">
    <div class="modal-header">
      <span class="modal-title">🎭 Select Tone</span>
      <button class="modal-close" data-close="tone">×</button>
    </div>

    <div class="variant-option selected" data-tone="curious">
      <div class="variant-radio"></div>
      <div class="variant-info">
        <div class="variant-name">🔍 Curious</div>
        <div class="variant-desc">Thoughtful and wisdom-seeking</div>
      </div>
    </div>

    <div class="variant-option" data-tone="playful">
      <div class="variant-radio"></div>
      <div class="variant-info">
        <div class="variant-name">🎮 Playful</div>
        <div class="variant-desc">Fun and lighthearted</div>
      </div>
    </div>

    <div class="variant-option" data-tone="serious">
      <div class="variant-radio"></div>
      <div class="variant-info">
        <div class="variant-name">📚 Serious</div>
        <div class="variant-desc">Structured and authoritative</div>
      </div>
    </div>
  </div>
</div>
```

### 3. State (line ~926)

```javascript
state.variants = {
  age: localStorage.getItem('kelly_age') || '18-35',
  language: localStorage.getItem('kelly_language') || 'en',
  tone: localStorage.getItem('kelly_tone') || 'curious', // ← ADD THIS
  difficulty: parseInt(localStorage.getItem('kelly_difficulty') || '2')
};
```

### 4. Update UI Function (line ~1121)

```javascript
function updateUI() {
  // ... existing code ...

  // Update tone badge
  const toneMap = { curious: 'C', playful: 'P', serious: 'S' };
  document.getElementById('badge-tone').textContent = toneMap[state.variants.tone];

  // Update modal selections
  document.querySelectorAll('[data-tone]').forEach((el) => {
    el.classList.toggle('selected', el.dataset.tone === state.variants.tone);
  });
}
```

### 5. Save Preferences (line ~1004)

```javascript
function savePreferences() {
  localStorage.setItem('kelly_age', state.variants.age);
  localStorage.setItem('kelly_language', state.variants.language);
  localStorage.setItem('kelly_tone', state.variants.tone); // ← ADD THIS
  localStorage.setItem('kelly_difficulty', state.variants.difficulty.toString());
}
```

### 6. Button Handler (line ~1487)

```javascript
document.getElementById('btn-tone').onclick = () => openModal('tone');
```

### 7. Option Handlers (line ~1518)

```javascript
document.querySelectorAll('[data-tone]').forEach((el) => {
  el.onclick = () => selectVariant('tone', el.dataset.tone);
});
```

---

## Notes

- Buttons are stacked vertically with 20px gap
- Icons are 28px inside 48px circles
- Badges are 18px tall, positioned top-right
- Labels are 10px font below icon
- Sound button is special (spinning animation)
- All buttons have glass morphism effect
- Active state scales down to 0.9







