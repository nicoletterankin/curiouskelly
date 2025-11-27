# Kelly 2D Avatar System

Clean, professional 2D avatar system using real Kelly images for the 5-phase Hot-or-Not learning journey.

## Overview

This system provides a smooth, elegant avatar experience with:
- **Real Kelly images** from the Best Character Reference folder
- **Crossfade transitions** between expressions
- **5-phase lesson flow** (Welcome → Q1 → Q2 → Q3 → Wisdom)
- **Hot or Not interactions** with immediate visual reactions
- **Minimal UI** that puts Kelly front and center

## Architecture

```
Kelly 2D System
├── kelly-2d-avatar.js     → Core avatar class
├── kelly-2d-avatar.css    → Clean professional styles
└── kelly-2d-demo.html     → Interactive demo
```

## Image Mapping

The system uses these Kelly reference images:

| Expression   | Image File | Use Case |
|-------------|------------|----------|
| Welcome     | `Curious Kelly in final pose in Chair - Copy.png` | Initial greeting |
| Question    | `facing to the left.png` | Asking questions |
| Explaining  | `neutral face with hair.png` | Teaching moments (Hot) |
| Celebrating | `head and shoulders without chair.png` | Reactions (Not) |
| Wisdom      | `head and shoulders without chair.png` | Final wisdom |

## API

### Initialize
```javascript
import { Kelly2DAvatar } from '/lesson-player/js/kelly-2d-avatar.js';

const container = document.getElementById('kelly-avatar-container');
const kelly = new Kelly2DAvatar(container);
```

### Methods
```javascript
// Phase transitions
kelly.showWelcome();
kelly.showQuestion(1);    // 1, 2, or 3
kelly.showReaction(1, 'a'); // question number, choice ('a' or 'b')
kelly.showWisdom();

// Direct phase control
kelly.setPhase('q1', 'a');
```

### Events
```javascript
document.addEventListener('kelly-phase-changed', (e) => {
  const { phase, expression } = e.detail;
  console.log(`Kelly: ${phase} → ${expression}`);
});
```

## Design Philosophy

**What This System IS:**
- Clean, professional, elegant
- Smooth crossfade transitions
- Real Kelly images
- Minimal, unobtrusive UI
- Fast and responsive

**What This System IS NOT:**
- Tacky effects or animations
- Generic stock photos
- Over-the-top UI elements
- Complex 3D rendering
- Resource-heavy

## Usage in Lesson Player

1. Include the CSS and JS:
```html
<link rel="stylesheet" href="/lesson-player/css/kelly-2d-avatar.css">
<script type="module" src="/lesson-player/js/kelly-2d-avatar.js"></script>
```

2. Create a container:
```html
<div id="kelly-avatar-container"></div>
```

3. Initialize and control:
```javascript
import { Kelly2DAvatar } from '/lesson-player/js/kelly-2d-avatar.js';

const kelly = new Kelly2DAvatar(
  document.getElementById('kelly-avatar-container')
);

// Use throughout lesson flow
kelly.showWelcome();
// ... lesson interactions ...
```

## Demo

Run the dev server:
```bash
cd daily-lesson-marketing
npm run dev
```

Visit: http://localhost:4321/lesson-player/kelly-2d-demo.html

## Future Enhancements

Once we have more Kelly images generated:
- **Age morphing** → Crossfade between age variants
- **Language switching** → Different poses for different languages
- **Tone variations** → Excited, curious, serene expressions
- **More reactions** → Surprised, thoughtful, encouraging
- **Custom animations** → Subtle movements (breathing, blinking)

## Asset Requirements

To extend the system, generate Kelly images with:
- **Consistent lighting** → Match existing reference photos
- **Same background** → Or clean, transparent PNG
- **High resolution** → At least 1920x1080
- **Named clearly** → `kelly-{expression}-{context}.png`

Place new images in: `/kelly-ref/Best Character Reference/`

Update image mapping in `kelly-2d-avatar.js` → `getImagePath()` method.

## Performance

- Images are preloaded for instant transitions
- Crossfades use CSS opacity (GPU-accelerated)
- Minimal DOM manipulation
- No runtime image processing
- Respects `prefers-reduced-motion`

## Accessibility

- Alt text on all images
- Keyboard navigation support (via parent controls)
- Reduced motion support
- High contrast state badges
- Screen reader friendly

---

Built with real Kelly images. Clean. Professional. Elegant.





