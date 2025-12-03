# Curious Kelly Diagnostic Report
**Generated:** Saturday, November 29, 2025

---

## Deployment Configuration

### Platform
- **Primary Platform:** Vercel
- **Backup Platform:** Netlify (configured but appears secondary)

### Config Files
| File | Path | Purpose |
|------|------|---------|
| vercel.json | `/vercel.json` | Root - serves `public/` folder directly |
| vercel.json | `/daily-lesson-marketing/vercel.json` | Astro build - outputs to `dist/` |
| netlify.toml | `/netlify.toml` | Backup - serves `public/` folder |
| netlify.toml | `/daily-lesson-marketing/netlify.toml` | (if exists) |

### Vercel Project
```json
{
  "projectId": "prj_UAQkFMEQDen9yEae2UFaPzNE1Lqq",
  "orgId": "team_UllshnsdJY8EDLsiuEGTOAfC",
  "projectName": "curiouskelly"
}
```

### Git Remotes
```
origin  https://github.com/nicoletterankin/curiouskelly.git (fetch)
origin  https://github.com/nicoletterankin/curiouskelly.git (push)
```

### Production Folders
| Folder | Purpose | Status |
|--------|---------|--------|
| `/public/` | Static files served by root `vercel.json` | ✅ Active |
| `/daily-lesson-marketing/dist/` | Astro build output | ✅ Built |
| `/daily-lesson-marketing/public/` | Astro source static files | ✅ Active |

### Key Files Location
| File | Path |
|------|------|
| learn.html | `/public/learn.html` |
| index.html (homepage) | `/public/index.html` |

---

## 2D Kelly Assets

### Director Chair Images (Primary Avatar Set)
**Location:** `/public/assets/kelly_canonical/core/chair/`

| File | Dimensions | Size |
|------|------------|------|
| kelly-chair-celebrating.png | 1408x768 | 1,155 KB |
| kelly-chair-curious.png | 1408x768 | 1,202 KB |
| kelly-chair-explaining.png | 1408x768 | 1,266 KB |
| kelly-chair-listening.png | 1408x768 | 1,200 KB |
| kelly-chair-wisdom.png | 1408x768 | 1,100 KB |

### Production WebP Avatars (Optimized Set)
**Location:** `/public/assets/kelly/production/avatars/`

| Expression | Sizes Available | 512px Size |
|------------|-----------------|------------|
| curious | 64, 128, 256, 512 (.jpg + .webp) | 12.1 KB |
| explaining | 64, 128, 256, 512 (.jpg + .webp) | 12.5 KB |
| listening | 64, 128, 256, 512 (.jpg + .webp) | 12.1 KB |
| wisdom | 64, 128, 256, 512 (.jpg + .webp) | 12.1 KB |
| celebrating | 64, 128, 256, 512 (.jpg + .webp) | 11.9 KB |

### Expression Images (Backup/Legacy)
**Location:** `/public/images/expressions/`

| File | Dimensions | Size |
|------|------------|------|
| celebrating.jpeg | 1024x1024 | 455 KB |
| confused.jpeg | 1024x1024 | 461 KB |
| curious-closeup.jpeg | 1024x1024 | 967 KB |
| curious-main.jpeg | 1024x1024 | 586 KB |
| curious-thinking.jpeg | 1024x1024 | 586 KB |
| explaining.jpeg | 1024x1024 | 494 KB |
| happy-content.jpeg | 1024x1024 | 4,725 KB |
| peaceful.jpeg | 1024x1024 | 451 KB |
| surprised.jpeg | 1024x1024 | 456 KB |

### Avatar Controller JS Files
| File | Path | Purpose |
|------|------|---------|
| kelly-2d-avatar.js | `/public/js/kelly-2d-avatar.js` | Main 2D avatar controller class |
| kelly-avatar-controller.js | `/public/js/kelly-avatar-controller.js` | Unified 2D/3D controller |
| unity-kelly-loader.js | `/public/js/unity-kelly-loader.js` | 3D Unity WebGL loader |

### Image Paths in Code
The `kelly-2d-avatar.js` expects images at:
```javascript
basePath: '/images/kelly/'  // Default
imageSet: 'directors-chair' // Default option
// Generates: /images/kelly/kelly-directors-chair-{expression}.png
```

**⚠️ ISSUE:** The actual chair images are at:
`/assets/kelly_canonical/core/chair/kelly-chair-{expression}.png`

---

## 3D Kelly / Unity Files

### Unity Project
**Location:** `/digital-kelly/engines/Kelly_Engine_V2/onlykelly/`

### Scene Files
| Scene | Path |
|-------|------|
| KellyMain.unity | `/Assets/Scenes/KellyMain.unity` |
| KellyMain.unity | `/Assets/KellyMain.unity` (duplicate?) |
| RL_PreviewScene.unity | `/Assets/Reallusion/CCiC Unity Tools/URP/Preview Scene/` |

### FBX Model Files
| File | Path |
|------|------|
| kelly_fbx_v4.fbx | `/onlykelly/Assets/Kelly/Animations/Lessons/` |
| Kelly_Live_v3.fbx | `/Assets/` |
| Kelly_Live_v2.fbx | `/Assets/` |
| Kelly_Live_v1.fbx | `/Assets/` |
| Kelly_Live_v2.fbx | `/onlykelly/Assets/` (copy) |
| Kelly_Live_v1.fbx | `/onlykelly/Assets/` (copy) |

### GLB Avatar Files (Age Variants)
**Location:** `/digital-kelly/content/balance/avatars/`

| File | Purpose |
|------|---------|
| kelly_avatar_age_3.glb | Toddler variant |
| kelly_avatar_age_9.glb | Child variant |
| kelly_avatar_age_15.glb | Teen variant |
| kelly_avatar_age_27.glb | Young adult variant |
| kelly_avatar_age_48.glb | Adult variant |
| kelly_avatar_age_82.glb | Elder variant |

### Character Creator Project
| File | Path |
|------|------|
| Kelly_Unity_Production.ccProject | `/digital-kelly/Kelly_Unity_Production.ccProject` |
| kelly_directors_chair.iProject | `/digital-kelly/kelly_directors_chair.iProject` |

### WebGL Build Output
**Location:** `/digital-kelly/engines/Kelly_Engine_V2/onlykelly/Kelly_Web_Build/`

| File | Type |
|------|------|
| Kelly_Web_Build.data.unityweb | Build data |
| Kelly_Web_Build.framework.js.unityweb | Framework |
| Kelly_Web_Build.loader.js | Loader script |
| Kelly_Web_Build.wasm.unityweb | WebAssembly |
| index.html | Entry point |

**Also copied to:**
- `/public/unity/kelly-v1/`
- `/public/unity/kelly-live/`
- `/daily-lesson-marketing/public/unity/kelly-v1/`
- `/daily-lesson-marketing/public/unity/kelly-live/`

### Unity Packages (66 total)
**Location:** `/digital-kelly/CCIC Auto Setup for Unity/Install in Unity/CCiC-Unity-Tools/Packages/`

Includes shader packages for:
- URP10, URP12, URP14, URP17, URP171, URP172
- HDRP10, HDRP12, HDRP14, HDRP17
- Built-In renderer

### Upwork Deliverables
**Location:** `/arif-deliveries/milestone-2-phase-1/`

| Folder | Status |
|--------|--------|
| original/ | **EMPTY** - Awaiting Arif's delivery |
| testing/ | Ready for test imports |
| screenshots/ | Ready for test screenshots |
| feedback/ | Ready for feedback |

**Expected deliverables:**
- .ccCharacter file with 52 morphs
- Separate L/R eye bones
- FBX export with blendshapes

---

## Supabase Configuration

### Connection Details
**Config Location:** `/public/config.js`

```javascript
window.SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
window.SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...';
```

**⚠️ WARNING:** Credentials are hardcoded in public JavaScript file.

### Tables Referenced
| Table | Purpose | Records |
|-------|---------|---------|
| `core_lessons` | 365 daily lessons | 365 rows |
| `lesson_atoms` | Content pieces per lesson | ~21,915 rows |
| `lesson_shards` | Demographic variants | ~38,700 rows |
| `users` | User profiles | Variable |
| `user_progress` | Progress tracking | Variable |

### Files Using Supabase
- `/public/config.js` - Main config
- `/public/js/kelly-data.js`
- `/public/js/auth.js`
- `/public/learn.html`
- `/public/app.html`
- `/app/supabase-service.js`
- `/daily-lesson-marketing/public/lesson-player/js/app.js`
- Multiple markdown documentation files

### Schema Documentation
- `/docs/backend/SUPABASE_SCHEMA.md` - Full schema reference
- `/docs/backend/SUPABASE_MCP_SETUP.md` - MCP setup guide

---

## Issues Found

### Critical Issues 🔴

1. **Image Path Mismatch**
   - Code expects: `/images/kelly/kelly-directors-chair-{expression}.png`
   - Actual path: `/assets/kelly_canonical/core/chair/kelly-chair-{expression}.png`
   - **Impact:** 2D avatar may show broken images

2. **Hardcoded Supabase Credentials**
   - Location: `/public/config.js`
   - **Risk:** Security concern - should use environment variables
   - **Note:** Anon key is public-safe but pattern is wrong

3. **Empty Upwork Deliverables**
   - Location: `/arif-deliveries/milestone-2-phase-1/original/`
   - **Status:** Awaiting delivery from Arif (Milestone 2)

### Medium Issues 🟡

4. **Duplicate Unity Builds**
   - Same build exists in multiple locations
   - `/public/unity/kelly-v1/`, `/public/unity/kelly-live/`
   - `/daily-lesson-marketing/public/unity/kelly-v1/`, etc.
   - **Impact:** Wasted storage, potential version confusion

5. **No .env Files Found**
   - Search returned 0 results for `.env*`
   - Only `/daily-lesson-marketing/env.template` exists
   - **Impact:** Secrets may be committed or not configured

6. **Dual Deployment Configs**
   - Both Vercel and Netlify are configured
   - Root `vercel.json` serves `/public/` directly
   - `daily-lesson-marketing/vercel.json` uses Astro build
   - **Question:** Which one serves curiouskelly.com?

### Minor Issues 🟢

7. **Large Expression Image**
   - `happy-content.jpeg` is 4.7MB (others are ~500KB)
   - **Impact:** Slow loading if used

8. **Duplicate KellyMain.unity**
   - Scene exists in both `/Assets/Scenes/` and `/Assets/`
   - **Impact:** Confusion about which is canonical

9. **Legacy Image Sets**
   - Multiple expression image sets exist
   - Chair images (1408x768)
   - Square expressions (1024x1024)
   - Production webp (64-512px)
   - **Question:** Which set is canonical?

---

## Architecture Summary

```
curiouskelly.com (Vercel)
├── / (root vercel.json)
│   └── serves: /public/
│       ├── index.html (homepage)
│       ├── learn.html (lesson player)
│       ├── js/ (avatar controllers)
│       └── assets/ (Kelly images)
│
├── /daily-lesson-marketing/ (Astro site)
│   ├── vercel.json (build config)
│   ├── src/ (Astro components)
│   └── dist/ (built output)
│
└── /digital-kelly/ (3D assets)
    ├── engines/Kelly_Engine_V2/ (Unity project)
    ├── content/ (GLB avatars, animations)
    └── *.ccProject (Character Creator files)
```

---

## Recommendations

1. **Fix image paths** in `kelly-2d-avatar.js` to match actual asset locations
2. **Move Supabase credentials** to environment variables
3. **Clarify which deployment** actually serves curiouskelly.com
4. **Deduplicate Unity builds** - keep one canonical location
5. **Create .env files** from `env.template`
6. **Document canonical image set** - chair vs expressions vs production webp

---

*Report complete. No changes have been made.*








