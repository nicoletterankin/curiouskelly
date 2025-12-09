# 16:9 OS Wallpaper Audit

## Core Objective
Ensure every interface component adheres to the "Curious Kelly OS" aesthetic: a fixed, immersive, desktop-like experience (16:9 aspect ratio preferred, or full viewport) rather than a traditional scrolling webpage.

## 1. Player OS (`player.html`)
**Status:** ✅ **Compliant**
- **Structure:** Fixed viewport (`100vh`, `100vw`, `overflow: hidden`).
- **Navigation:** Uses a "Drawer" and "Modals" (Tuition, Reader, Settings) to keep the user within the context.
- **Links:** 
  - `Syllabus`, `Careers`, `Newsroom`, `Tuition` all open internal modals.
  - `Logout` redirects to `index.html`.
- **Recommendations:**
  - Ensure `Logout` triggers a "Shut Down" animation before redirecting to maintain immersion.

## 2. Marketing Portal (`index.html`)
**Status:** ❌ **Non-Compliant**
- **Current State:** Standard vertical scrolling landing page.
- **Violations:**
  - Page scrolls vertically (standard web behavior).
  - Footer is below the fold.
  - Content flows linearly rather than spatially.
- **Fix Plan:**
  - **Convert to "Login Screen" OS Mode:**
    - Lock viewport to 100vh/100vw.
    - Keep the Split Screen (Left: Login/Info, Right: Kelly/Wallpaper).
    - **Syllabus/Tuition/Features:** Move these from scrolling sections into "Desktop Icons" or "Menu Items" that open "Windows" (Modals) just like the Player.
    - **Footer:** Convert to a fixed bottom "Taskbar" or "Status Bar" with small triggers for "About", "Privacy", etc.

## 3. Content Pages (`about.html`, `careers.html`, `newsroom.html`, `enterprise.html`)
**Status:** ❌ **Non-Compliant**
- **Current State:** Traditional long-form scrolling webpages.
- **Violations:**
  - Full page scroll.
  - Standard headers/footers breaks the "App" illusion.
- **Fix Plan:**
  - **Option A (Ideal for OS feel):** Deprecate these as standalone pages and merge their content into `index.html` (for public) and `player.html` (for authenticated) as **Modals**.
  - **Option B (SEO Friendly):** Apply a "Window Wrapper" CSS.
    - The `<body>` becomes the wallpaper background.
    - The content is wrapped in a `.os-window` container centered on screen (aspect ratio 16:9 or max-height).
    - Scroll happens *inside* this window, not on the body.
    - Header becomes the "Window Title Bar".

## 4. Detailed Link Audit
| Page | Link | Destination | Compliance Check | Action |
|------|------|-------------|------------------|--------|
| `index.html` | Login | `player.html` | ✅ | Good transition to OS. |
| `index.html` | Footer Links | `about.html`, etc. | ❌ | Breaks immersion (opens standard page). |
| `player.html` | Drawer > Syllabus | `openContentModal` | ✅ | Opens internal modal. |
| `player.html` | Drawer > Careers | `openContentModal` | ✅ | Opens internal modal (Mock content). |
| `player.html` | Drawer > Newsroom | `openContentModal` | ✅ | Opens internal modal (Mock content). |
| `about.html` | Header > Home | `index.html` | ❌ | Standard navigation. |

## Executive Summary
To achieve the "OS Wallpaper" feel, we must eliminate "Web Page Scrolling" entirely. The Public Site (`index.html`) should function as the **"Lock Screen"** or **"Public Desktop"**, where information (Pricing, About, Careers) opens in **Windows** over the wallpaper, exactly how `player.html` handles its content.

**Next Steps:**
1. Create a shared `os-layout.css` that enforces fixed body background.
2. Refactor `index.html` to be a non-scrolling "Portal".
3. Convert `about`, `careers`, etc., into "Content Windows" overlaying the portal.

















