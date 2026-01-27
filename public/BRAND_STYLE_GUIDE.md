# CURIOUS KELLY BRAND STYLE GUIDE
Generated: 01/14/2026 11:48:49
SOURCE OF TRUTH - DO NOT MODIFY WITHOUT FOUNDER APPROVAL

## ╔══════════════════════════════════════════════════════════╗
## ║           CURIOUS KELLY BRAND STYLE GUIDE                ║
## ╠══════════════════════════════════════════════════════════╣
## ║                                                          ║
## ║  LOGO: Kelly's face (NEVER letters)                      ║
## ║                                                          ║
## ║  PRIMARY COLOR: BLUE #3B82F6                             ║
## ║  SECONDARY: Dark Blue #2563EB                            ║
## ║  ACCENT: Light Blue #60A5FA                              ║
## ║                                                          ║
## ║  FORBIDDEN:                                              ║
## ║  ❌ Orange (#f97316)                                     ║
## ║  ❌ Amber (#f59e0b)                                      ║
## ║  ❌ "K" or "CK" logos                                    ║
## ║  ❌ Stock photos as Kelly                                ║
## ║                                                          ║
## ║  BACKGROUND: #030712 (near black)                        ║
## ║  SURFACE: #18181B (zinc-900)                             ║
## ║  TEXT: #FFFFFF                                           ║
## ║                                                          ║
## ╚══════════════════════════════════════════════════════════╝

## COLOR PALETTE

### Primary Colors (USE THESE)
| Name | Hex | Tailwind | Usage |
|------|-----|----------|-------|
| Primary Blue | #3B82F6 | blue-500 | Buttons, links, accents |
| Dark Blue | #2563EB | blue-600 | Hover states, emphasis |
| Light Blue | #60A5FA | blue-400 | Highlights, secondary |

### Neutral Colors (USE THESE)
| Name | Hex | Tailwind | Usage |
|------|-----|----------|-------|
| Background | #030712 | gray-950 | Main background |
| Surface | #18181B | zinc-900 | Cards, panels |
| Border | #27272A | zinc-800 | Dividers |
| Text Primary | #FFFFFF | white | Main text |
| Text Secondary | #A1A1AA | zinc-400 | Muted text |

### Semantic Colors (USE THESE)
| Name | Hex | Tailwind | Usage |
|------|-----|----------|-------|
| Success | #22C55E | green-500 | Positive states |
| Error | #EF4444 | red-500 | Error states |
| Warning | #3B82F6 | blue-500 | Use blue, NOT orange |

## FORBIDDEN COLORS ❌

NEVER use these colors in Curious Kelly brand:

| Color | Hex | Why Forbidden |
|-------|-----|---------------|
| Orange-500 | #f97316 | Not part of brand |
| Orange-400 | #fb923c | Not part of brand |
| Orange-600 | #ea580c | Not part of brand |
| Amber-500 | #f59e0b | Not part of brand |
| Amber-600 | #d97706 | Not part of brand |
| Amber-400 | #fbbf24 | Not part of brand |

## EXCEPTION: Warning/Alert States

For warnings in emails and alerts, use BLUE (#3B82F6) instead of orange.
If absolutely necessary for contrast, use a muted blue tone.

## LOGO RULES

| Asset | ✅ CORRECT | ❌ WRONG |
|-------|-----------|----------|
| Logo | Kelly's face | "K", "CK", text |
| Favicon | Kelly's face (cropped) | Letter icon |
| App Icon | Kelly's face | Abstract shape |
| Header | Kelly's face + optional text | Text only |
| Brand Mark | Kelly's face | Stylized anything |

## FILES TO UPDATE

The following files contain orange/amber violations and should be updated:

### Critical (Core UI)
- components/apps/git-bash.tsx
- components/apps/notes.tsx  
- components/apps/finder.tsx
- components/apps/safari.tsx
- TEMPLATES/v0/ArchetypeCard.tsx
- TEMPLATES/v0/LessonPreviewCard.tsx
- TEMPLATES/v0/lib/personas.ts

### CSS Files
- public/css/brand-colors.css
- public/css/brand-tokens.css
- public/css/kelly-magic.css
- public/css/kelly-os.css
- public/css/phase-commons.css
- lessons/unified-app.css

### API/Backend
- api/email/*.ts (email templates use gold for accents)
- lib/email/escalation.ts
- lib/visual-prompts.ts

