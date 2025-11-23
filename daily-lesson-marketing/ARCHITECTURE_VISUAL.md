# 📊 System Architecture Visual Guide

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER BROWSER                          │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   HTML/CSS    │  │  JavaScript  │  │   Assets     │    │
│  │  (Static)     │  │  (Islands)    │  │  (Images)    │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
                         ↑           ↓
                    Requests    Submits Form
                         ↓           ↑
┌─────────────────────────────────────────────────────────────┐
│                     CDN / Hosting                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │          Static Files (HTML, CSS, JS)                │   │
│  │          Served from CDN edge locations              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                         ↑
                    API Requests
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              Serverless Functions                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  /api/lead                                             │  │
│  │  1. Verify Turnstile                                   │  │
│  │  2. Validate data                                      │  │
│  │  3. Forward to CRM                                     │  │
│  │  4. Return response                                    │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                         ↓
                    Webhook POST
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    CRM System                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │          Stores lead data                            │   │
│  │          Sends confirmation                          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Build Process Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    SOURCE CODE                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   .astro     │  │   .ts/.tsx   │  │    .scss     │     │
│  │   files      │  │   files      │  │    files     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                         ↓
                    npm run build
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    ASTRO BUILD                              │
│                                                              │
│  1. Parse .astro files                                      │
│     ├─ Execute frontmatter (server-side)                    │
│     ├─ Render HTML template                                 │
│     └─ Extract <script> and <style>                        │
│                                                              │
│  2. Compile TypeScript                                      │
│     ├─ TypeScript → JavaScript                              │
│     └─ Bundle with Vite                                     │
│                                                              │
│  3. Process Styles                                          │
│     ├─ SCSS → CSS                                           │
│     └─ Extract critical CSS                                 │
│                                                              │
│  4. Optimize Images                                         │
│     ├─ Generate WebP/AVIF                                  │
│     └─ Create responsive srcsets                            │
│                                                              │
│  5. Generate Output                                         │
│     ├─ Static HTML files                                    │
│     ├─ JavaScript bundles                                   │
│     ├─ CSS files                                            │
│     └─ Serverless function code                             │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                      DIST FOLDER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   index.html │  │   _assets/   │  │   .functions │     │
│  │   about.html │  │   (JS/CSS)   │  │   (API)     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                         ↓
                    Deploy to Platform
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              PRODUCTION (CDN + Serverless)                  │
└─────────────────────────────────────────────────────────────┘
```

## Component Hierarchy

```
SiteLayout.astro
│
├── Header
│   ├── Navbar
│   │   ├── Logo
│   │   ├── Nav Items (Home, Adults, Children, etc.)
│   │   └── LanguageSwitcher (dropdown)
│   └── Mobile Menu (hamburger)
│
├── Main Content (varies by page)
│   │
│   ├── Home Page (index.astro)
│   │   ├── Hero Section
│   │   │   ├── HeroCountdown (island)
│   │   │   └── CTA Button
│   │   ├── Trust Badges
│   │   ├── Features Grid
│   │   ├── Pricing Card
│   │   ├── LeadForm (island - interactive)
│   │   ├── TestimonialsCarousel (island - Slick)
│   │   └── FAQ Accordion
│   │
│   ├── Other Pages
│   │   ├── Adults, Children, Companies
│   │   └── Privacy, Cookies, Thank You
│   │
│   └── API Routes
│       ├── /api/lead (serverless function)
│       └── /api/rum (serverless function)
│
└── Footer
    ├── Links
    ├── Legal (Privacy, Cookies)
    └── Store Badges (App Store, Google Play)

CookieConsent (overlay - island)
```

## Data Flow: Form Submission

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERACTION                          │
│                                                              │
│  1. User fills form fields                                  │
│  2. Types email, phone, etc.                                │
│  3. Selects country (triggers region dropdown)               │
│  4. Checks marketing opt-in                                 │
│  5. Interacts with Turnstile widget                         │
│  6. Clicks "Submit" button                                  │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              CLIENT-SIDE VALIDATION                         │
│  ┌────────────────────────────────────────────────────┐   │
│  │  validateLeadForm()                                 │   │
│  │  ├─ Check required fields                           │   │
│  │  ├─ Validate email format (RFC)                    │   │
│  │  ├─ Validate phone (E.164)                         │   │
│  │  ├─ Check name length/charset                       │   │
│  │  └─ Show errors inline if invalid                   │   │
│  └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                         ↓ (if valid)
┌─────────────────────────────────────────────────────────────┐
│              TURNSTILE VERIFICATION                          │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Widget generates token                              │   │
│  │  Token included in submission                        │   │
│  └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    HTTP POST                                │
│  POST /api/lead                                             │
│  Headers: Content-Type: application/json                    │
│  Body: {                                                     │
│    first_name: "John",                                      │
│    last_name: "Doe",                                        │
│    email: "john@example.com",                              │
│    phone: "+1234567890",                                   │
│    country: "US",                                          │
│    region: "CA",                                           │
│    marketing_opt_in: false,                                │
│    turnstile_token: "0.abc123..."                          │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              SERVERLESS FUNCTION                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │  1. Parse request body                               │   │
│  │  2. Verify Turnstile token                           │   │
│  │     └─ POST to Cloudflare API                        │   │
│  │  3. Server-side validation                           │   │
│  │     └─ validateLeadForm()                           │   │
│  │  4. Sanitize data                                    │   │
│  │     └─ sanitizeFormData()                           │   │
│  │  5. Forward to CRM webhook                           │   │
│  │     └─ POST to CRM_WEBHOOK_URL                       │   │
│  │  6. Return JSON response                             │   │
│  └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                         ↓                    ↓
              Response to Client      Webhook to CRM
                         ↓                    ↓
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT HANDLES                           │
│                                                              │
│  If success:                                                │
│    ├─ Track analytics event                                │
│    └─ Redirect to /thank-you                               │
│                                                              │
│  If error:                                                  │
│    ├─ Show error message                                   │
│    └─ Allow retry                                          │
└─────────────────────────────────────────────────────────────┘
```

## i18n Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    URL DETECTION                             │
│                                                              │
│  User visits: /es-es/                                       │
│       ↓                                                      │
│  getLocaleFromPath('/es-es/') → 'es-ES'                     │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  GET TRANSLATIONS                             │
│                                                              │
│  getTranslations('es-ES')                                    │
│       ↓                                                      │
│  Returns: esES object                                       │
│  {                                                           │
│    hero: { headline: "Domina el Inglés..." },               │
│    form: { title: "Comienza Hoy", ... },                    │
│    ...                                                       │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  RENDER PAGE                                 │
│                                                              │
│  Page component receives locale                              │
│       ↓                                                      │
│  Components use translations:                                │
│  <h1>{t.hero.headline}</h1>                                  │
│       ↓                                                      │
│  Output: "Domina el Inglés..."                              │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    SEO TAGS                                  │
│                                                              │
│  Generate hreflang tags:                                     │
│  <link hreflang="es-ES" href="/es-es/" />                  │
│  <link hreflang="en-US" href="/" />                        │
│  <link hreflang="pt-BR" href="/pt-br/" />                 │
└─────────────────────────────────────────────────────────────┘
```

## Consent Management Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    FIRST VISIT                               │
│                                                              │
│  Page loads → CookieConsent component                        │
│       ↓                                                      │
│  Check localStorage for 'consentState'                       │
│       ↓                                                      │
│  Not found → Show consent banner                            │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  USER CHOOSES                                │
│                                                              │
│  Option 1: Accept All                                        │
│    └─ Set: { necessary: true, analytics: true,              │
│            marketing: true }                                 │
│                                                              │
│  Option 2: Reject All                                       │
│    └─ Set: { necessary: true, analytics: false,            │
│            marketing: false }                               │
│                                                              │
│  Option 3: Customize                                        │
│    └─ User toggles individual options                       │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    STORE STATE                               │
│                                                              │
│  localStorage.setItem('consentState',                        │
│    JSON.stringify({ ... }))                                 │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              TRIGGER SCRIPT LOADING                         │
│                                                              │
│  consentManager.onChange() fires                             │
│       ↓                                                      │
│  If analytics: true                                         │
│    └─ loadGTM()                                             │
│    └─ loadGA4()                                             │
│                                                              │
│  If marketing: true                                          │
│    └─ loadMetaPixel()                                        │
│    └─ loadTikTokPixel()                                     │
│    └─ loadTwitterPixel()                                    │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  SUBSEQUENT VISITS                           │
│                                                              │
│  Check localStorage                                         │
│       ↓                                                      │
│  Found → Load scripts based on stored state                  │
│       ↓                                                      │
│  No banner shown (already consented)                         │
└─────────────────────────────────────────────────────────────┘
```

## Testing Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      UNIT TESTS                              │
│  (Vitest - Fast, Isolated)                                   │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  validation  │  │  countdown   │  │  sanitize    │    │
│  │  tests       │  │  tests       │  │  tests       │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                              │
│  Tests individual functions                                  │
│  Runs in milliseconds                                        │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    E2E TESTS                                 │
│  (Playwright - Full Browser)                                 │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Form submit │  │  Consent      │  │  Language    │    │
│  │  flow        │  │  manager      │  │  switching   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                              │
│  Tests user flows end-to-end                                 │
│  Runs in seconds                                             │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                 PERFORMANCE TESTS                            │
│  (Lighthouse CI)                                             │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  LCP < 2.5s  │  │  CLS < 0.1   │  │  INP < 200ms │    │
│  │  Check       │  │  Check       │  │  Check       │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                              │
│  Validates performance budgets                                │
│  Runs in CI/CD pipeline                                      │
└─────────────────────────────────────────────────────────────┘
```

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GIT REPOSITORY                            │
│                                                              │
│  Code pushed to main branch                                  │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                 GITHUB ACTIONS                               │
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │  1. Checkout code                                   │   │
│  │  2. Install dependencies                            │   │
│  │  3. Run tests                                       │   │
│  │  4. Build project                                   │   │
│  │  5. Deploy based on DEPLOY_TARGET                  │   │
│  └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                         ↓
        ┌────────────────┴────────────────┐
        ↓                ↓                ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   VERCEL     │  │   NETLIFY     │  │  CLOUDFLARE  │
│              │  │               │  │               │
│  Uploads    │  │  Uploads      │  │  Uploads      │
│  /dist      │  │  /dist        │  │  /dist        │
│              │  │               │  │               │
│  Serves     │  │  Serves       │  │  Serves       │
│  via CDN    │  │  via CDN      │  │  via CDN      │
│              │  │               │  │               │
│  Functions  │  │  Functions    │  │  Workers      │
│  on Edge    │  │  on Edge      │  │  on Edge      │
└──────────────┘  └──────────────┘  └──────────────┘
        ↓                ↓                ↓
┌─────────────────────────────────────────────────────────────┐
│                      LIVE SITE                                │
│                                                              │
│  Static files served from CDN                                │
│  Serverless functions handle API calls                        │
│  Global edge network for fast delivery                       │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts Visualization

### Static Site Generation vs Server-Side Rendering

```
STATIC (This System):
┌─────────┐      Build Time      ┌─────────┐
│  Code   │ ────→ Generate HTML ──→ │   HTML  │
└─────────┘                       └─────────┘
                                           │
                                    User Request
                                           ↓
                                    Serve instantly
                                    (No processing)

SERVER-SIDE RENDERING:
┌─────────┐      Every Request    ┌─────────┐
│  Code   │ ────→ Generate HTML ──→ │   HTML  │
└─────────┘                       └─────────┘
     ↑                                  │
     └────── User Request ──────────────┘
     (Slower, requires server)
```

### Component Islands Concept

```
TRADITIONAL SPA (All JavaScript):
┌─────────────────────────────────────────┐
│  Entire page is JavaScript               │
│  ┌─────────────────────────────────────┐ │
│  │  Heavy JS bundle (300KB+)           │ │
│  │  Hydrates everything                │ │
│  │  Slower initial load                │ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘

ASTRO ISLANDS (This System):
┌─────────────────────────────────────────┐
│  Mostly static HTML                      │
│  ┌────────┐  ┌────────┐  ┌────────┐  │
│  │ Static │  │ Island  │  │ Static │  │
│  │  HTML  │  │  JS     │  │  HTML  │  │
│  └────────┘  └────────┘  └────────┘  │
│  Fast load    Only needed  Fast load   │
└─────────────────────────────────────────┘
         ↓                ↓
    No JS needed    Hydrates only
    for static      interactive parts
```

---

This visual guide complements the detailed TEACHING_GUIDE.md. Use both together to understand the system!


















