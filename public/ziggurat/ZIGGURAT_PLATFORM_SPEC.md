# Ziggurat LED Vision Platform

## Final Specification Document

**Version:** 1.0  
**Date:** January 2026  
**Status:** Ready for Implementation

---

## 1. Executive Summary

Transform the Ziggurat LED Vision concept into a stakeholder-facing web platform that enables:
- Public viewing of before/after visualizations
- Stakeholder feedback collection
- Admin control over content and settings
- Progress tracking toward implementation

**Core Deliverable:** A single deployable web application with public site + admin panel.

---

## 2. User Types

| Role | Access | Capabilities |
|------|--------|--------------|
| **Public** | Unauthenticated | View visualizations, interactive slider, download assets |
| **Stakeholder** | Email-verified | Submit feedback, vote on options, view project updates |
| **Admin** | Password + 2FA | Manage content, view analytics, export data, configure settings |

---

## 3. Public Site Features

### 3.1 Landing Page (`/`)
- Hero section with project title and tagline
- Animated before/after comparison (auto-sliding on load)
- Call-to-action: "Explore the Vision"

### 3.2 Vision Gallery (`/vision`)
- Interactive before/after slider (drag to compare)
- Time-of-day variants: Night / Dusk / Day
- Color palette options (if multiple designs)
- Full-screen mode
- Download buttons for each asset

### 3.3 Technical Specs (`/specs`)
- Building dimensions and terrace geometry
- LED specifications (linear feet, color capabilities)
- Detected terrace coordinates visualization
- Dark-sky compliance notes
- Sustainability considerations

### 3.4 Feedback (`/feedback`)
- Stakeholder registration (email verification)
- Survey form:
  - Preference voting (which variant?)
  - Open-ended comments
  - Support level (1-5 scale)
- View aggregate results (after submission)

### 3.5 Updates (`/updates`)
- Project timeline
- News posts from admin
- Milestone tracker

### 3.6 About (`/about`)
- Project background
- Team / organization
- Contact information
- Press kit download

---

## 4. Admin Panel Features

### 4.1 Dashboard (`/admin`)
- Total views, unique visitors
- Feedback submissions count
- Average support rating
- Recent activity feed

### 4.2 Content Management (`/admin/content`)
- Upload/replace visualization images
- Edit page text (WYSIWYG)
- Manage news posts
- Configure downloadable assets

### 4.3 Feedback Management (`/admin/feedback`)
- View all submissions (table with search/filter)
- Export to CSV
- Flag/archive responses
- Aggregate statistics

### 4.4 Stakeholders (`/admin/stakeholders`)
- View registered stakeholders
- Manual add/remove
- Send email updates (integration with Resend/Postmark)

### 4.5 Settings (`/admin/settings`)
- Site title and metadata
- Enable/disable feedback collection
- Configure color palette options
- API keys management
- Backup/export all data

### 4.6 Analytics (`/admin/analytics`)
- Page views over time
- Asset download counts
- Feedback submission trends
- Geographic distribution (if collected)

---

## 5. Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│  Next.js 14 (App Router) + Tailwind CSS + shadcn/ui         │
│  - Public pages (SSG for performance)                        │
│  - Admin panel (client-side with auth)                       │
│  - Interactive slider component                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Layer                               │
│  Next.js API Routes or Vercel Edge Functions                 │
│  - /api/feedback (POST submissions)                          │
│  - /api/admin/* (protected routes)                           │
│  - /api/analytics (view tracking)                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Database                                │
│  Supabase (PostgreSQL + Auth + Storage)                      │
│  - feedback table                                            │
│  - stakeholders table                                        │
│  - content table (CMS)                                       │
│  - analytics_events table                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Asset Storage                            │
│  Supabase Storage or Cloudflare R2                           │
│  - /images/before/*.jpg                                      │
│  - /images/after/*.jpg                                       │
│  - /downloads/*.zip                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Database Schema

### 6.1 `feedback`
```sql
CREATE TABLE feedback (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT now(),
  email TEXT,
  name TEXT,
  variant_preference TEXT,  -- 'night' | 'dusk' | 'day'
  support_level INT CHECK (support_level BETWEEN 1 AND 5),
  comments TEXT,
  ip_hash TEXT,  -- anonymized
  archived BOOLEAN DEFAULT false
);
```

### 6.2 `stakeholders`
```sql
CREATE TABLE stakeholders (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT now(),
  email TEXT UNIQUE NOT NULL,
  name TEXT,
  organization TEXT,
  verified BOOLEAN DEFAULT false,
  role TEXT DEFAULT 'stakeholder'
);
```

### 6.3 `content`
```sql
CREATE TABLE content (
  id TEXT PRIMARY KEY,  -- 'hero_title', 'about_text', etc.
  value JSONB NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT now(),
  updated_by UUID REFERENCES stakeholders(id)
);
```

### 6.4 `posts`
```sql
CREATE TABLE posts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT now(),
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  published BOOLEAN DEFAULT false,
  author_id UUID REFERENCES stakeholders(id)
);
```

### 6.5 `analytics_events`
```sql
CREATE TABLE analytics_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT now(),
  event_type TEXT NOT NULL,  -- 'page_view', 'download', 'slider_interact'
  page TEXT,
  metadata JSONB,
  session_id TEXT
);
```

---

## 7. API Endpoints

### Public
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/content/:id` | Fetch content block |
| GET | `/api/posts` | List published posts |
| POST | `/api/feedback` | Submit feedback (rate limited) |
| POST | `/api/analytics` | Track event |

### Protected (Admin)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/admin/feedback` | List all feedback |
| GET | `/api/admin/feedback/export` | CSV export |
| PATCH | `/api/admin/feedback/:id` | Archive/flag |
| GET | `/api/admin/stakeholders` | List stakeholders |
| POST | `/api/admin/stakeholders` | Add stakeholder |
| PUT | `/api/admin/content/:id` | Update content |
| GET | `/api/admin/analytics/summary` | Dashboard stats |

---

## 8. Key Components

### 8.1 Before/After Slider
```tsx
// components/BeforeAfterSlider.tsx
interface Props {
  beforeSrc: string;
  afterSrc: string;
  initialPosition?: number; // 0-100
}
```
- Touch and mouse support
- Keyboard accessible (arrow keys)
- Full-screen toggle
- Download button overlay

### 8.2 Variant Selector
```tsx
// components/VariantSelector.tsx
type Variant = 'night' | 'dusk' | 'day';
interface Props {
  variants: { id: Variant; label: string; thumbnail: string }[];
  selected: Variant;
  onChange: (v: Variant) => void;
}
```

### 8.3 Feedback Form
```tsx
// components/FeedbackForm.tsx
interface FeedbackData {
  email: string;
  name?: string;
  variantPreference: Variant;
  supportLevel: 1 | 2 | 3 | 4 | 5;
  comments?: string;
}
```
- Client-side validation
- Rate limiting feedback
- Success confirmation with aggregate preview

---

## 9. Deployment

### Hosting
- **Vercel** (Next.js optimized, edge functions, analytics)
- Domain: `ziggurat.vision` or subdomain of existing property

### Environment Variables
```env
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=
ADMIN_PASSWORD_HASH=
RESEND_API_KEY=  # for email
```

### CI/CD
- GitHub Actions on push to `main`
- Preview deployments for PRs
- Database migrations via Supabase CLI

---

## 10. Assets Required

### From Current Work
| Asset | Source | Status |
|-------|--------|--------|
| `aerial.jpg` | Original photo | ✅ Have |
| `HERO-before-full.png` | Rendered | ✅ Have |
| `HERO-night-full.png` | Rendered | ✅ Have |
| `after-night-*.jpg` | Rendered | ✅ Have |
| `after-dusk-*.jpg` | Rendered | ✅ Have |
| `after-day-*.jpg` | Rendered | ✅ Have |
| `comparison-*.jpg` | Rendered | ✅ Have |

### To Create
| Asset | Description |
|-------|-------------|
| Logo | Ziggurat Vision wordmark |
| Favicon | Pyramid icon |
| OG Image | Social share preview |
| Press kit | ZIP with all assets + fact sheet |

---

## 11. Implementation Phases

### Phase 1: Static Site (1-2 days)
- [ ] Deploy current `pitch-assets/index.html` to Vercel
- [ ] Add proper meta tags and OG image
- [ ] Configure custom domain
- [ ] Basic analytics (Vercel Analytics)

### Phase 2: Next.js Migration (2-3 days)
- [ ] Convert to Next.js App Router
- [ ] Component library setup (shadcn/ui)
- [ ] Responsive design polish
- [ ] All public pages

### Phase 3: Database + Auth (1-2 days)
- [ ] Supabase project setup
- [ ] Database schema creation
- [ ] Admin authentication
- [ ] Protected routes

### Phase 4: Feedback System (1-2 days)
- [ ] Feedback form component
- [ ] API endpoint with validation
- [ ] Admin feedback viewer
- [ ] Export functionality

### Phase 5: CMS + Polish (2-3 days)
- [ ] Content management for editable text
- [ ] News/updates system
- [ ] Email integration for stakeholder updates
- [ ] Final QA and launch

**Total estimated: 7-12 days**

---

## 12. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Page views | 1,000+ in first month | Vercel Analytics |
| Feedback submissions | 50+ responses | Database count |
| Average support rating | 4.0+ / 5.0 | Calculated average |
| Asset downloads | 100+ | Event tracking |
| Stakeholder signups | 25+ verified | Database count |

---

## 13. Open Questions

1. **Domain:** Use new domain or subdomain of existing property?
2. **Branding:** Need logo/visual identity or use text-only?
3. **Email:** Which service for stakeholder communications?
4. **Privacy:** GDPR compliance requirements?
5. **Timeline:** Hard deadline for launch?

---

## 14. Files in This Repository

```
public/ziggurat/
├── pitch-assets/           # Generated stakeholder assets
│   ├── index.html          # Current static pitch page
│   ├── HERO-*.png          # Full-res hero images
│   ├── after-*-*.jpg       # Variant renders
│   ├── before-*.jpg        # Original at resolutions
│   └── comparison-*.jpg    # Side-by-side
├── render_pitch.py         # Asset generation script
├── render_enhanced.py      # Edge-detection renderer
├── aerial.jpg              # Source photograph
├── ZIGGURAT_PLATFORM_SPEC.md  # This document
└── LAYER_ARCHITECTURE.md   # Technical LED placement docs
```

---

## Next Action

**To proceed, confirm:**
1. Domain/hosting preference
2. Phase 1 launch target date
3. Whether to use existing Supabase project or create dedicated

Ready to build.
