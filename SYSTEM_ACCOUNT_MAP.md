## Curious Kelly Accounts & Architecture – Current vs Future

### 1. Source of Truth
- **GitHub:** `nicoletterankin/curiouskelly` (main branch auto-deploys)
- **Local repo:** `C:\Users\user\UI-TARS-desktop`

---

### 2. Account Inventory (Current State)
| Layer | Provider | Account / Project | Purpose | Notes |
| --- | --- | --- | --- | --- |
| DNS / CDN | **Cloudflare** | `curiouskelly.com` zone | DNS + proxy + email records | `A` record → `76.76.21.21` (Vercel). Cloudflare Pages still hosts legacy lesson players (`curiouskelly-lessons`, `curiouskelly-lessons-v2`). |
| Hosting (prod) | **Vercel / Lotd org** | Project `curiouskelly-1mv5` | Public site (`public/` folder) | Deploys from GitHub `main`. No build step, static export only. |
| Hosting (legacy) | **Vercel** | `daily-lesson-marketing` (Astro) | Deprecated marketing site | Still contains Astro project + build config (`daily-lesson-marketing/vercel.json`). Not mapped to domain. |
| Hosting (legacy) | **Cloudflare Pages** | `curiouskelly-lessons` & `curiouskelly-lessons-v2` | Old lesson player (`lesson-player/index.html`) | Taking requests but should be retired; new Unity player coming. |
| Auth + DB | **Supabase** | Project `tvjalxxsyryjphkforjv` | Auth, database, storage | SQL schema lives in `supabase-schema-clean.sql`. OAuth redirect: `https://tvjalxxsyryjphkforjv.supabase.co/auth/v1/callback`. |
| OAuth Provider | **Google Cloud** | Project `gen-lang-client-0005524332` | Google OAuth + Vertex keys | OAuth client `Curious Kelly Production`. API keys labeled `antigravity`, `api key 1`, `reinmaker`, `Kelly CCS`. |
| Automation / Generators | **ANTIGRAVITY workspace** | VS Code + Python scripts | Bulk content + persona generation | Writes to `/antigravity` folder; uses same Supabase DB via `.env`. |
| Voice / Media APIs | **ElevenLabs, HeyGen, etc.** | Keys stored in repo `.env` templates | Generate lesson audio/video | Governed by `CLAUDE.md` and voice pipeline docs. |

---

### 3. Current Build (Third & Fourth Images)
The “University of Curiosity” experience now lives in two static entry points:

1. `public/index.html` – **Portal / Login**  
   - Supabase OAuth buttons (Google, Apple, GitHub)  
   - Terms + privacy links  
   - Kelly hero image and scribbles (Claude-style)  

2. `public/about.html` – **Campus / Daily Lesson Showcase**  
   - “Today’s Lesson” hero, Unity player placeholder  
   - Curriculum tracks, syllabus calendar, tuition cards  
   - Footer linking to all auxiliary pages (careers, newsroom, etc.)

These two pages are the entire externally visible system today. Everything else (dashboard, service layer, Unity build, affiliate tools) is either static copy or under construction.

---

### 4. Current Architecture Diagram
```
Browser
 ├─ curiouskelly.com → Vercel (public/index.html + about.html + footer pages)
 ├─ curiouskelly.com/dashboard.html (static shell; awaits real data)
 └─ curiouskelly.com/unity/... (WebGL build placeholder)

Vercel project curiouskelly-1mv5
 └─ Serves /public with no build step

Cloudflare DNS
 ├─ A @ → 76.76.21.21 (Vercel edge)
 ├─ CNAME www → cname.vercel-dns.com
 └─ Legacy Pages sites (curiouskelly-lessons, curiouskelly-lessons-v2) still active

Supabase (tvjalxxsyryjphkforjv)
 ├─ Auth providers enabled (Google in progress, Apple/GitHub pending)
 ├─ SQL schema defined (profiles, lessons, user_progress, affiliates, etc.)
 └─ Storage buckets planned for images/audio

Google Cloud (gen-lang-client-0005524332)
 ├─ OAuth client “Curious Kelly Production”
 └─ API keys for Antigravity / Reinmaker generators
```

---

### 5. Gaps / Risks
1. **Domain confusion:** Cloudflare Pages still serves `curiouskelly-lessons*.pages.dev`; old assets may leak if DNS rewrites are added later.
2. **OAuth partially configured:** Google client created but Apple/GitHub not finished; Supabase settings must match.
3. **No backend service layer:** Dashboard/auth flows rely on Supabase client only; no Render/Railway API yet.
4. **Unity player pending:** Lesson player referenced in `about.html` does not exist yet—Unity build lives under `public/unity/kelly-v1/`.
5. **Automation overlap:** Antigravity scripts write to same Supabase but lack environment isolation.

---

### 6. Future-State Target (Mapping Third/Fourth Images to System)
| Layer | Future Owner | Description |
| --- | --- | --- |
| **Access Portal (`index.html`)** | Vercel static → Supabase-auth | Keep current design; connect buttons to Google/Apple/GitHub; auto-redirect to dashboard when authenticated. |
| **Campus (`about.html`)** | Vercel static + live syllabus API | Maintain Claude aesthetic; pull “Today’s Lesson” + live calendar data from Supabase via API to avoid hardcoded copy. |
| **Lesson Player** | Unity WebGL (`public/unity/kelly-v1/`) + service layer | Replace legacy HTML player with Unity build; load lesson manifests + audio from Supabase storage/CDN. |
| **Service Layer API** | Render/Railway Node service (`api.curiouskelly.com`) | Implements lesson, user, affiliate, enterprise endpoints described in `PRODUCTION_ARCHITECTURE.md`. |
| **Data & Auth** | Supabase | Single source for users, lessons, affiliates, enterprise leads, newsletter signups. Enforce RLS + triggers from `supabase-schema-clean.sql`. |
| **Automation (Antigravity, Reinmaker)** | Cloud scripts hitting Supabase | Continue generating PhaseDNA + assets, but write through service layer to keep data consistent. |

---

### 7. Migration Checklist
1. **Accounts**
   - Disable Cloudflare Pages deployments (`curiouskelly-lessons*`) after Unity player ships.
   - Keep only one Vercel project (`curiouskelly-1mv5`) attached to domain.
2. **Auth**
   - Finish Google OAuth (save client) → Enable Apple/GitHub in Supabase.
   - Update `public/js/auth.js` with secure env injection (no keys inline).
3. **Backend**
   - Deploy Supabase schema (`supabase-schema-clean.sql`), verify RLS.
   - Stand up API on Render/Railway, point `api.curiouskelly.com` via Cloudflare.
4. **Frontend**
   - Wire `index.html` to real auth; add session check → `dashboard.html`.
   - Connect `about.html` sections (lesson, calendar, tuition) to live data.
   - Integrate Unity player iframe with service layer.
5. **Decommission Legacy**
   - Archive `daily-lesson-marketing/` Astro site (docs only).
   - Remove `deployment/vercel.json.OLD` once confident.

---

### 8. Open Questions
1. Do we keep any Cloudflare Workers around the lesson player, or move everything to Vercel?
2. How will Antigravity pipelines authenticate (service account vs Supabase key)?
3. Where do we host heavy assets (Supabase storage vs Cloudflare R2) once Unity build needs >100 MB?

---

### 9. Next Actions
1. Finish Google OAuth save + test login on `curiouskelly.com`.
2. Enable remaining providers, update `auth.js` to pull keys from environment.
3. Deploy Supabase schema and seed lessons.
4. Build minimal Node API (Render) that wraps Supabase for dashboard.
5. Replace Cloudflare Pages lesson player with Unity build referenced in `about.html`.






























