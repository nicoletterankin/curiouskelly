# Publishing to v0 Team Workspace

## Overview

v0's team templates are created by:
1. Creating a component in v0 chat
2. Saving it as a template
3. It appears in the "From Lotd" tab

## Step-by-Step Publishing Guide

### Step 1: Open v0

1. Go to https://v0.app
2. Ensure you're logged in as **Lotd** team
3. Click "New Chat"

### Step 2: Publish Each Template

For each template, paste the code and ask v0 to understand it:

#### Template 1: ArchetypeCard

```
I want to save this as a team template called "ArchetypeCard".

[Paste contents of ArchetypeCard.tsx]

Please understand this component. It's an adaptive card that styles itself based on 12 Kelly teaching archetypes. Each archetype has a unique color, icon, and tagline.
```

Then click the "..." menu on the generated component and select "Save as Template".

#### Template 2: LessonPreviewCard

```
I want to save this as a team template called "LessonPreviewCard".

[Paste contents of LessonPreviewCard.tsx]

This is a marketing card with 4 variants: default, compact, hero, and social. It can fetch lesson data from Supabase or accept props directly.
```

#### Template 3: FactoryDayView

```
I want to save this as a team template called "FactoryDayView".

[Paste contents of FactoryDayView.tsx]

This is a 12×5 grid showing all 12 archetypes × 5 phases for a single lesson day. It connects to Supabase to show content/video generation status.
```

### Step 3: Verify Templates

1. Go to https://v0.app/templates/team
2. Click "From Lotd" tab
3. Verify all templates appear:
   - ArchetypeCard
   - LessonPreviewCard
   - FactoryDayView

### Step 4: Test Template Usage

Start a new chat and try:

```
Using my ArchetypeCard template, create a grid of all 12 archetypes showing their lesson counts for this week.
```

v0 should recognize and use your template.

---

## Template Naming Convention

| Template File | v0 Template Name |
|--------------|------------------|
| `ArchetypeCard.tsx` | ArchetypeCard |
| `LessonPreviewCard.tsx` | LessonPreviewCard |
| `FactoryDayView.tsx` | FactoryDayView |

---

## Alternative: Design System Upload

If v0 supports Design System upload:

1. Go to https://v0.app/design-systems
2. Click "Create Design System"
3. Name it "Lotd Components"
4. Upload the `templates/v0/` directory

---

## Verification Checklist

- [ ] ArchetypeCard template saved
- [ ] LessonPreviewCard template saved
- [ ] FactoryDayView template saved
- [ ] Templates visible in "From Lotd" tab
- [ ] Test prompt works with templates

---

## Direct Links (After Publishing)

Templates will be available at:
- https://v0.app/templates/team (From Lotd tab)

Share with team:
```
🎨 v0 Templates Ready!

Our team templates are now live in v0. Open v0.app → Templates → From Lotd to use:
• ArchetypeCard - Adaptive cards for all 12 archetypes
• LessonPreviewCard - Marketing cards (4 variants)
• FactoryDayView - 12×5 production grid

Example prompt: "Using my ArchetypeCard template, create a lesson picker for Day 17"
```


















