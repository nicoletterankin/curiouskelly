# Ziggurat LED Vision — v0.app Handoff Package

## Instructions for v0

You are taking over development of the Ziggurat LED Vision web platform. This document contains everything you need to build a stakeholder-facing Next.js application.

**Your mission:** Create a polished, production-ready Next.js 14 App Router application with:
1. Public site for viewing LED vision mockups
2. Admin panel for managing feedback and content
3. Interactive before/after comparison components

---

## Project Context

### What is this?
A visualization project for adding LED lighting to the Chet Holifield Federal Building (the "Ziggurat") in Laguna Niguel, California. The building is a stepped pyramid (ziggurat shape) with 7 terrace levels.

### What exists?
- Real aerial photograph of the building (4032×2217 pixels)
- Rendered mockups showing rainbow LED bands on each terrace
- Edge-detection algorithm that found the exact Y positions of terrace seams
- Multiple time-of-day variants (night, dusk, day)

### What we're building?
A web platform where stakeholders can:
- View before/after visualizations with interactive slider
- Submit feedback on which variant they prefer
- Download high-res assets
- Track project updates

---

## Design System

### Colors
```tsx
const colors = {
  // Rainbow LED palette (purple top to red bottom)
  led: {
    level7: '#8B5CF6', // Purple - crown
    level6: '#3B82F6', // Blue
    level5: '#06B6D4', // Cyan
    level4: '#22C55E', // Green
    level3: '#EAB308', // Yellow
    level2: '#F97316', // Orange
    level1: '#EF4444', // Red - base
  },
  
  // UI colors
  background: '#0A0A0C',
  surface: '#111114',
  surfaceHover: '#1A1A1E',
  border: '#27272A',
  text: '#FFFFFF',
  textMuted: '#888888',
  textSubtle: '#555555',
  
  // Accent
  primary: '#8B5CF6',
  primaryHover: '#7C3AED',
}
```

### Typography
```tsx
const typography = {
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  
  hero: {
    fontSize: 'clamp(2.5rem, 6vw, 5rem)',
    fontWeight: 200,
    letterSpacing: '-0.02em',
  },
  
  h1: { fontSize: '2.5rem', fontWeight: 300 },
  h2: { fontSize: '2rem', fontWeight: 300 },
  h3: { fontSize: '1.2rem', fontWeight: 500 },
  
  body: { fontSize: '1rem', lineHeight: 1.6 },
  small: { fontSize: '0.875rem' },
}
```

### Spacing
```tsx
const spacing = {
  section: '80px',
  gap: '24px',
  padding: '40px',
  borderRadius: {
    sm: '8px',
    md: '12px',
    lg: '16px',
  }
}
```

---

## Components Required

### 1. BeforeAfterSlider

Interactive comparison slider for before/after images.

```tsx
interface BeforeAfterSliderProps {
  beforeSrc: string;
  afterSrc: string;
  beforeAlt?: string;
  afterAlt?: string;
  initialPosition?: number; // 0-100, default 50
  showLabels?: boolean;
  className?: string;
}

// Features:
// - Drag handle with ◀ ▶ indicator
// - Touch support
// - Keyboard accessible (arrow keys)
// - Smooth animation
// - Full-screen toggle button
// - Download buttons for both images
```

**Implementation notes:**
- Use CSS `clip-path: inset()` for the reveal effect
- Handle should be 4px white vertical line with centered pill button
- Labels "BEFORE" and "AFTER" in top corners with semi-transparent background

### 2. VariantSelector

Thumbnail grid for selecting time-of-day variants.

```tsx
type Variant = 'night' | 'dusk' | 'day';

interface VariantSelectorProps {
  variants: {
    id: Variant;
    label: string;
    description: string;
    thumbnailSrc: string;
  }[];
  selected: Variant;
  onChange: (variant: Variant) => void;
}

// Variants data:
const variants = [
  { id: 'night', label: 'Night', description: 'Full rainbow spectrum', thumbnailSrc: '/ziggurat/pitch-assets/after-night-1080p.jpg' },
  { id: 'dusk', label: 'Dusk', description: 'Warm transition', thumbnailSrc: '/ziggurat/pitch-assets/after-dusk-1080p.jpg' },
  { id: 'day', label: 'Day', description: 'Subtle presence', thumbnailSrc: '/ziggurat/pitch-assets/after-day-1080p.jpg' },
];
```

### 3. FeedbackForm

Stakeholder feedback collection form.

```tsx
interface FeedbackFormData {
  email: string;
  name?: string;
  organization?: string;
  variantPreference: 'night' | 'dusk' | 'day' | 'none';
  supportLevel: 1 | 2 | 3 | 4 | 5;
  comments?: string;
}

// Features:
// - Client-side validation
// - Star rating component for supportLevel
// - Radio buttons for variant preference
// - Success state with aggregate preview
// - Rate limiting (1 submission per email per day)
```

### 4. SpecsCard

Technical specification display cards.

```tsx
interface SpecsCardProps {
  value: string | number;
  label: string;
  icon?: React.ReactNode;
}

// Usage:
<SpecsCard value="7" label="Terrace Levels" />
<SpecsCard value="~2,100" label="Linear Feet of LED" />
<SpecsCard value="Rainbow" label="Color Spectrum" />
<SpecsCard value="Edge-Aligned" label="Placement Method" />
```

### 5. DownloadButton

Asset download button with size indicator.

```tsx
interface DownloadButtonProps {
  href: string;
  filename: string;
  label: string;
  size?: string; // e.g., "6.4 MB"
  variant?: 'primary' | 'secondary';
}
```

### 6. TimelineItem

Project milestone timeline component.

```tsx
interface TimelineItemProps {
  date: string;
  title: string;
  description: string;
  status: 'completed' | 'current' | 'upcoming';
}
```

### 7. AdminLayout

Admin panel layout with sidebar navigation.

```tsx
interface AdminLayoutProps {
  children: React.ReactNode;
}

// Sidebar items:
const adminNav = [
  { href: '/admin', label: 'Dashboard', icon: HomeIcon },
  { href: '/admin/feedback', label: 'Feedback', icon: MessageIcon },
  { href: '/admin/stakeholders', label: 'Stakeholders', icon: UsersIcon },
  { href: '/admin/content', label: 'Content', icon: FileTextIcon },
  { href: '/admin/analytics', label: 'Analytics', icon: ChartIcon },
  { href: '/admin/settings', label: 'Settings', icon: SettingsIcon },
];
```

---

## Page Structure

### Public Routes

```
/                     # Landing page with hero and CTA
/vision               # Main visualization gallery with slider
/specs                # Technical specifications
/feedback             # Stakeholder feedback form
/updates              # Project news and timeline
/about                # About the project
```

### Admin Routes

```
/admin                # Dashboard with stats
/admin/feedback       # View/export feedback
/admin/stakeholders   # Manage stakeholder list
/admin/content        # Edit page content
/admin/analytics      # View analytics
/admin/settings       # Site configuration
```

---

## Page Specifications

### Landing Page (`/`)

```tsx
// Structure:
<main>
  <HeroSection>
    <GradientTitle>Ziggurat LED Vision</GradientTitle>
    <Subtitle>Transforming the Chet Holifield Federal Building into a beacon of civic innovation</Subtitle>
    <CTAButton href="/vision">Explore the Vision</CTAButton>
  </HeroSection>
  
  <PreviewSection>
    <BeforeAfterSlider 
      beforeSrc="/ziggurat/pitch-assets/before-1080p.jpg"
      afterSrc="/ziggurat/pitch-assets/after-night-1080p.jpg"
      initialPosition={50}
    />
  </PreviewSection>
  
  <StatsSection>
    <SpecsCard value="7" label="Terrace Levels" />
    <SpecsCard value="~2,100" label="Linear Feet" />
    <SpecsCard value="Rainbow" label="Spectrum" />
  </StatsSection>
</main>
```

**Hero gradient text CSS:**
```css
.gradient-title {
  background: linear-gradient(135deg, #8b5cf6 0%, #06b6d4 50%, #22c55e 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
```

### Vision Gallery (`/vision`)

```tsx
<main>
  <PageHeader title="Vision Gallery" />
  
  <VariantSelector 
    variants={variants}
    selected={selectedVariant}
    onChange={setSelectedVariant}
  />
  
  <BeforeAfterSlider
    beforeSrc="/ziggurat/pitch-assets/before-1080p.jpg"
    afterSrc={`/ziggurat/pitch-assets/after-${selectedVariant}-1080p.jpg`}
  />
  
  <DownloadSection>
    <DownloadButton 
      href={`/ziggurat/pitch-assets/after-${selectedVariant}-4k.jpg`}
      label="Download 4K"
      size="~1 MB"
    />
    <DownloadButton 
      href="/ziggurat/pitch-assets/HERO-night-full.png"
      label="Download Full Resolution"
      size="6.4 MB"
      variant="primary"
    />
  </DownloadSection>
</main>
```

### Feedback Page (`/feedback`)

```tsx
<main>
  <PageHeader 
    title="Share Your Feedback" 
    subtitle="Help shape the future of this civic landmark"
  />
  
  <FeedbackForm onSubmit={handleSubmit} />
  
  {submitted && (
    <ResultsPreview>
      <h3>Community Response</h3>
      <PieChart data={variantPreferences} />
      <AverageRating value={averageSupportLevel} />
    </ResultsPreview>
  )}
</main>
```

---

## Database Schema (Supabase)

### Tables

```sql
-- Feedback submissions
CREATE TABLE feedback (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT now(),
  email TEXT NOT NULL,
  name TEXT,
  organization TEXT,
  variant_preference TEXT CHECK (variant_preference IN ('night', 'dusk', 'day', 'none')),
  support_level INT CHECK (support_level BETWEEN 1 AND 5),
  comments TEXT,
  ip_hash TEXT,
  archived BOOLEAN DEFAULT false
);

-- Stakeholder registry
CREATE TABLE stakeholders (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT now(),
  email TEXT UNIQUE NOT NULL,
  name TEXT,
  organization TEXT,
  verified BOOLEAN DEFAULT false,
  role TEXT DEFAULT 'stakeholder' CHECK (role IN ('stakeholder', 'admin'))
);

-- CMS content blocks
CREATE TABLE content (
  id TEXT PRIMARY KEY,
  value JSONB NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT now(),
  updated_by UUID REFERENCES stakeholders(id)
);

-- Project updates/news
CREATE TABLE posts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT now(),
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  published BOOLEAN DEFAULT false,
  publish_date TIMESTAMPTZ,
  author_id UUID REFERENCES stakeholders(id)
);

-- Analytics events
CREATE TABLE analytics_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT now(),
  event_type TEXT NOT NULL,
  page TEXT,
  metadata JSONB,
  session_id TEXT
);

-- Row Level Security
ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE stakeholders ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Anyone can insert feedback" ON feedback FOR INSERT WITH CHECK (true);
CREATE POLICY "Only admins can view feedback" ON feedback FOR SELECT USING (
  auth.jwt() ->> 'role' = 'admin'
);
```

---

## API Routes

### `/api/feedback` (POST)

```tsx
// app/api/feedback/route.ts
import { createClient } from '@supabase/supabase-js';
import { NextResponse } from 'next/server';
import { z } from 'zod';

const feedbackSchema = z.object({
  email: z.string().email(),
  name: z.string().optional(),
  organization: z.string().optional(),
  variantPreference: z.enum(['night', 'dusk', 'day', 'none']),
  supportLevel: z.number().min(1).max(5),
  comments: z.string().max(2000).optional(),
});

export async function POST(request: Request) {
  const body = await request.json();
  const data = feedbackSchema.parse(body);
  
  // Rate limiting check
  // Insert to Supabase
  // Return success
}
```

### `/api/feedback/aggregate` (GET)

```tsx
// Returns aggregate statistics for public display
interface AggregateResponse {
  totalResponses: number;
  averageSupportLevel: number;
  variantPreferences: {
    night: number;
    dusk: number;
    day: number;
    none: number;
  };
}
```

### `/api/admin/feedback` (GET) — Protected

```tsx
// Requires admin authentication
// Returns paginated feedback list with filters
interface FeedbackListResponse {
  data: Feedback[];
  pagination: {
    page: number;
    pageSize: number;
    total: number;
  };
}
```

---

## Asset Locations

All assets are in `/public/ziggurat/`:

### Primary Assets (use these)
```
/ziggurat/pitch-assets/
├── index.html                    # Static pitch page (reference)
├── HERO-before-full.png          # 6.4 MB - Full res before
├── HERO-night-full.png           # 6.4 MB - Full res night
├── before-1080p.jpg              # 460 KB - Web before
├── before-4k.jpg                 # 1.2 MB - 4K before
├── after-night-1080p.jpg         # 293 KB - Web night
├── after-night-4k.jpg            # 843 KB - 4K night
├── after-dusk-1080p.jpg          # 410 KB - Web dusk
├── after-dusk-4k.jpg             # 1.1 MB - 4K dusk
├── after-day-1080p.jpg           # 464 KB - Web day
├── after-day-4k.jpg              # 1.3 MB - 4K day
├── comparison-night.jpg          # 763 KB - Side-by-side
└── comparison-dusk.jpg           # 880 KB - Side-by-side
```

### Source Photo
```
/ziggurat/aerial.jpg              # 1.3 MB - Original 4032×2217
```

---

## Technical Specifications (for /specs page)

```tsx
const specifications = {
  building: {
    name: 'Chet Holifield Federal Building',
    nickname: 'The Ziggurat',
    location: 'Laguna Niguel, California',
    architect: 'William Pereira',
    completed: 1971,
    floors: 12,
    terraces: 7,
  },
  
  led: {
    levels: 7,
    linearFeet: 2100, // approximate
    colorCapability: 'Full RGB',
    spectrum: 'Rainbow (Purple → Red)',
    controlSystem: 'DMX/Art-Net compatible',
  },
  
  // Detected terrace coordinates (Y pixels in 4032×2217 image)
  terraceEdges: {
    level7: { y: 1222, normalized: 0.551 }, // Crown (purple)
    level6: { y: 1264, normalized: 0.570 },
    level5: { y: 1336, normalized: 0.602 },
    level4: { y: 1393, normalized: 0.628 },
    level3: { y: 1435, normalized: 0.647 },
    level2: { y: 1492, normalized: 0.673 },
    level1: { y: 1540, normalized: 0.694 }, // Base (red)
  },
  
  imageSource: {
    resolution: '4032 × 2217 pixels',
    aspectRatio: '1.82:1',
    format: 'JPEG',
  },
};
```

---

## Environment Variables

```env
# Supabase
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_ROLE_KEY=eyJ...

# Admin auth (simple password for MVP)
ADMIN_PASSWORD_HASH=xxx

# Optional: Email service
RESEND_API_KEY=re_xxx

# Optional: Analytics
NEXT_PUBLIC_VERCEL_ANALYTICS_ID=xxx
```

---

## File Structure

```
app/
├── layout.tsx                    # Root layout with dark theme
├── page.tsx                      # Landing page
├── vision/
│   └── page.tsx                  # Vision gallery
├── specs/
│   └── page.tsx                  # Technical specs
├── feedback/
│   └── page.tsx                  # Feedback form
├── updates/
│   └── page.tsx                  # News/timeline
├── about/
│   └── page.tsx                  # About page
├── admin/
│   ├── layout.tsx                # Admin layout with sidebar
│   ├── page.tsx                  # Dashboard
│   ├── feedback/
│   │   └── page.tsx              # Feedback list
│   ├── stakeholders/
│   │   └── page.tsx              # Stakeholder management
│   ├── content/
│   │   └── page.tsx              # CMS
│   ├── analytics/
│   │   └── page.tsx              # Analytics
│   └── settings/
│       └── page.tsx              # Settings
└── api/
    ├── feedback/
    │   ├── route.ts              # POST feedback
    │   └── aggregate/
    │       └── route.ts          # GET aggregate
    └── admin/
        ├── feedback/
        │   └── route.ts          # GET/PATCH feedback
        └── stakeholders/
            └── route.ts          # CRUD stakeholders

components/
├── ui/                           # shadcn/ui components
├── BeforeAfterSlider.tsx
├── VariantSelector.tsx
├── FeedbackForm.tsx
├── SpecsCard.tsx
├── DownloadButton.tsx
├── TimelineItem.tsx
├── AdminLayout.tsx
├── AdminSidebar.tsx
├── DataTable.tsx
└── StarRating.tsx

lib/
├── supabase.ts                   # Supabase client
├── utils.ts                      # Utility functions
└── constants.ts                  # Shared constants

public/
└── ziggurat/                     # All assets (see Asset Locations)
```

---

## Key Implementation Details

### Dark Theme Setup

```tsx
// app/layout.tsx
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#0A0A0C] text-white antialiased">
        {children}
      </body>
    </html>
  );
}
```

### Tailwind Config Extensions

```js
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        background: '#0A0A0C',
        surface: '#111114',
        'surface-hover': '#1A1A1E',
        border: '#27272A',
      },
      animation: {
        'bounce-slow': 'bounce 2s infinite',
      },
    },
  },
};
```

### Before/After Slider Logic

```tsx
// Key implementation for the slider
function BeforeAfterSlider({ beforeSrc, afterSrc, initialPosition = 50 }) {
  const [position, setPosition] = useState(initialPosition);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);

  const handleMove = (clientX: number) => {
    if (!containerRef.current || !isDragging.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setPosition(pct);
  };

  return (
    <div 
      ref={containerRef}
      className="relative overflow-hidden rounded-xl cursor-ew-resize"
      onMouseDown={() => isDragging.current = true}
      onMouseUp={() => isDragging.current = false}
      onMouseMove={(e) => handleMove(e.clientX)}
      onTouchMove={(e) => handleMove(e.touches[0].clientX)}
    >
      {/* After image (background) */}
      <img src={afterSrc} className="w-full" />
      
      {/* Before image (clipped) */}
      <img 
        src={beforeSrc}
        className="absolute inset-0 w-full h-full object-cover"
        style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
      />
      
      {/* Slider handle */}
      <div 
        className="absolute top-0 h-full w-1 bg-white"
        style={{ left: `${position}%`, transform: 'translateX(-50%)' }}
      >
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-white text-black px-3 py-2 rounded-full text-sm font-semibold whitespace-nowrap">
          ◀ ▶
        </div>
      </div>
      
      {/* Labels */}
      <span className="absolute top-4 left-4 bg-black/80 px-4 py-2 rounded-lg text-sm font-semibold">
        BEFORE
      </span>
      <span className="absolute top-4 right-4 bg-black/80 px-4 py-2 rounded-lg text-sm font-semibold">
        AFTER
      </span>
    </div>
  );
}
```

---

## Success Criteria

The platform is complete when:

1. ✅ Public site loads fast (<2s LCP)
2. ✅ Before/after slider works on mobile and desktop
3. ✅ All 3 variants (night/dusk/day) are viewable
4. ✅ Feedback form submits to database
5. ✅ Admin can view and export feedback
6. ✅ Assets downloadable at multiple resolutions
7. ✅ Proper meta tags and OG image for sharing
8. ✅ Accessible (keyboard navigation, screen readers)
9. ✅ Responsive design (mobile-first)

---

## Go Build

You have everything you need:
- Design system ✅
- Component specs ✅
- Page structure ✅
- Database schema ✅
- API routes ✅
- Asset locations ✅
- Technical details ✅

Start with the BeforeAfterSlider component and landing page. The assets are ready in `/public/ziggurat/pitch-assets/`.

**First prompt to v0:**
> "Create a Next.js 14 App Router application with a dark theme landing page for the Ziggurat LED Vision project. Include a hero section with gradient text, an interactive before/after image comparison slider, and a section showing 4 specification cards. Use the design system and component specs from the handoff document."
