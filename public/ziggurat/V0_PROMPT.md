# v0.app Prompt — Copy This Directly

## Prompt 1: Landing Page with Slider

```
Create a Next.js 14 App Router landing page for "Ziggurat LED Vision" with:

1. HERO SECTION:
- Dark background (#0A0A0C)
- Large gradient text title: "Ziggurat LED Vision" 
  (gradient: purple #8B5CF6 → cyan #06B6D4 → green #22C55E)
- Subtitle: "Transforming the Chet Holifield Federal Building into a beacon of civic innovation"
- Font: system font, weight 200, size clamp(2.5rem, 6vw, 5rem)

2. BEFORE/AFTER SLIDER:
- Interactive comparison slider with drag handle
- Before image: "/ziggurat/pitch-assets/before-1080p.jpg"
- After image: "/ziggurat/pitch-assets/after-night-1080p.jpg"
- White vertical line handle with centered "◀ ▶" pill button
- "BEFORE" and "AFTER" labels in top corners (black/80 background)
- Touch and mouse support
- CSS clip-path for reveal effect

3. SPECS CARDS:
- 4 cards in a row: "7 Terrace Levels", "~2,100 Linear Feet", "Rainbow Spectrum", "Edge-Aligned"
- Dark card background (#111114)
- Large value, small label below

4. CTA BUTTON:
- "Explore the Vision" → links to /vision
- Purple background (#8B5CF6), rounded-lg

Style: shadcn/ui, Tailwind CSS, fully dark theme, modern and minimal.
```

---

## Prompt 2: Vision Gallery Page

```
Create a Next.js vision gallery page at /vision with:

1. PAGE HEADER:
- Title: "Vision Gallery"
- Dark theme continuing from landing

2. VARIANT SELECTOR:
- 3 thumbnail cards in a row: Night, Dusk, Day
- Each shows preview image and label
- Selected state has purple border
- Images:
  - Night: "/ziggurat/pitch-assets/after-night-1080p.jpg"
  - Dusk: "/ziggurat/pitch-assets/after-dusk-1080p.jpg"
  - Day: "/ziggurat/pitch-assets/after-day-1080p.jpg"

3. MAIN SLIDER:
- Same BeforeAfterSlider component
- Before: "/ziggurat/pitch-assets/before-1080p.jpg"
- After: dynamically changes based on selected variant

4. DOWNLOAD SECTION:
- Download buttons for 4K and Full resolution
- Show file size (e.g., "1.2 MB")
- Primary button style for main download

5. STATE MANAGEMENT:
- useState for selected variant
- Default to 'night'

Include hover effects on variant cards and smooth transitions.
```

---

## Prompt 3: Feedback Form

```
Create a feedback form page at /feedback with:

1. FORM FIELDS:
- Email (required, validation)
- Name (optional)
- Organization (optional)
- Variant preference (radio: Night / Dusk / Day / No preference)
- Support level (1-5 star rating, clickable stars)
- Comments (textarea, max 2000 chars)
- Submit button

2. STAR RATING COMPONENT:
- 5 clickable stars
- Filled stars are yellow (#EAB308)
- Empty stars are gray (#555)
- Hover preview effect

3. VALIDATION:
- Client-side with react-hook-form or similar
- Show errors inline

4. SUBMIT:
- POST to /api/feedback
- Show loading state
- Show success message with aggregate preview

5. AGGREGATE PREVIEW (after submit):
- "X responses so far"
- Average support rating
- Simple bar chart of variant preferences

Dark theme, accessible, mobile-friendly.
```

---

## Prompt 4: Admin Dashboard

```
Create an admin dashboard at /admin with:

1. LAYOUT:
- Sidebar navigation on left (collapsible on mobile)
- Main content area on right
- Dark theme

2. SIDEBAR ITEMS:
- Dashboard (home icon) → /admin
- Feedback (message icon) → /admin/feedback
- Stakeholders (users icon) → /admin/stakeholders
- Content (file icon) → /admin/content
- Analytics (chart icon) → /admin/analytics
- Settings (gear icon) → /admin/settings

3. DASHBOARD CONTENT:
- 4 stat cards in grid:
  - Total Page Views
  - Feedback Submissions
  - Average Rating (with stars)
  - Asset Downloads
- Recent activity feed (list of recent events)

4. STAT CARD COMPONENT:
- Large number value
- Small label
- Optional trend indicator (+12% this week)
- Icon

Use shadcn/ui Card, use Lucide icons, keep consistent dark styling.
```

---

## Prompt 5: Feedback Admin Table

```
Create a feedback management page at /admin/feedback with:

1. DATA TABLE:
- Columns: Date, Email, Name, Variant, Rating, Actions
- Sortable columns
- Pagination (10 per page)
- Search/filter by email

2. ROW ACTIONS:
- View full details (modal or slide-over)
- Archive/unarchive toggle
- Delete (with confirmation)

3. BULK ACTIONS:
- Select all checkbox
- Export selected to CSV
- Archive selected

4. FILTERS:
- Date range picker
- Variant filter dropdown
- Rating filter (min stars)
- Archived toggle

5. EMPTY STATE:
- "No feedback yet" with illustration

Use shadcn/ui Table, Dialog, DropdownMenu. Server component with client islands for interactivity.
```

---

## Component: BeforeAfterSlider (Full Implementation)

```tsx
"use client";

import { useState, useRef, useCallback } from "react";
import Image from "next/image";

interface BeforeAfterSliderProps {
  beforeSrc: string;
  afterSrc: string;
  initialPosition?: number;
}

export function BeforeAfterSlider({ 
  beforeSrc, 
  afterSrc, 
  initialPosition = 50 
}: BeforeAfterSliderProps) {
  const [position, setPosition] = useState(initialPosition);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);

  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setPosition(pct);
  }, []);

  const handleMouseDown = () => { isDragging.current = true; };
  const handleMouseUp = () => { isDragging.current = false; };
  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging.current) handleMove(e.clientX);
  };
  const handleTouchMove = (e: React.TouchEvent) => {
    handleMove(e.touches[0].clientX);
  };

  return (
    <div
      ref={containerRef}
      className="relative overflow-hidden rounded-2xl cursor-ew-resize select-none shadow-2xl"
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onMouseMove={handleMouseMove}
      onTouchStart={handleMouseDown}
      onTouchEnd={handleMouseUp}
      onTouchMove={handleTouchMove}
    >
      {/* After image (full, background) */}
      <Image
        src={afterSrc}
        alt="After"
        width={1920}
        height={1055}
        className="w-full h-auto"
        priority
      />

      {/* Before image (clipped) */}
      <div
        className="absolute inset-0"
        style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
      >
        <Image
          src={beforeSrc}
          alt="Before"
          width={1920}
          height={1055}
          className="w-full h-full object-cover"
          priority
        />
      </div>

      {/* Slider handle */}
      <div
        className="absolute top-0 h-full w-1 bg-white shadow-lg"
        style={{ left: `${position}%`, transform: "translateX(-50%)" }}
      >
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-white text-black px-4 py-2 rounded-full text-sm font-bold whitespace-nowrap shadow-xl">
          ◀ ▶
        </div>
      </div>

      {/* Labels */}
      <span className="absolute top-6 left-6 bg-black/80 backdrop-blur-sm px-5 py-2.5 rounded-lg text-sm font-semibold tracking-wide">
        BEFORE
      </span>
      <span className="absolute top-6 right-6 bg-black/80 backdrop-blur-sm px-5 py-2.5 rounded-lg text-sm font-semibold tracking-wide">
        AFTER
      </span>
    </div>
  );
}
```

---

## Database Seed Data

```sql
-- Initial content blocks
INSERT INTO content (id, value) VALUES
('hero_title', '"Ziggurat LED Vision"'),
('hero_subtitle', '"Transforming the Chet Holifield Federal Building into a beacon of civic innovation"'),
('about_text', '{"paragraphs": ["The Chet Holifield Federal Building...", "Our vision..."]}');

-- Sample admin user
INSERT INTO stakeholders (email, name, role, verified) VALUES
('admin@ziggurat.vision', 'Admin', 'admin', true);
```

---

## Quick Reference

| What | Where |
|------|-------|
| Before 1080p | `/ziggurat/pitch-assets/before-1080p.jpg` |
| Night 1080p | `/ziggurat/pitch-assets/after-night-1080p.jpg` |
| Dusk 1080p | `/ziggurat/pitch-assets/after-dusk-1080p.jpg` |
| Day 1080p | `/ziggurat/pitch-assets/after-day-1080p.jpg` |
| Hero Full | `/ziggurat/pitch-assets/HERO-night-full.png` |
| Background | `#0A0A0C` |
| Surface | `#111114` |
| Primary | `#8B5CF6` |
| LED Colors | Purple→Blue→Cyan→Green→Yellow→Orange→Red |

---

## Start Here

Copy Prompt 1 into v0.app to generate the landing page. Then iterate with the other prompts for each page.
