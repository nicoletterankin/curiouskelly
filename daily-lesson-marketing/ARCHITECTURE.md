# Architecture Documentation

## Overview

The Daily Lesson marketing site is a Jamstack application built with Astro, designed for high performance, SEO, and lead generation. It supports multiple deployment targets (Vercel, Netlify, Cloudflare Pages) and includes comprehensive internationalization.

## Tech Stack

- **Framework**: Astro 4.0+ (static output with hybrid rendering)
- **Language**: TypeScript
- **Styling**: Bootstrap 4.6 + SCSS
- **Build**: Vite (Astro default)
- **Testing**: Vitest (unit), Playwright (E2E), Lighthouse CI (performance)

## Architecture Decisions

### Static Site Generation

- All pages are pre-rendered at build time
- Serverless functions handle dynamic form submissions
- Partial hydration via Astro Islands for interactive components

### Internationalization Strategy

- URL-based i18n: `/` (en-US), `/es-es/`, `/pt-br/`
- Translation dictionaries in `src/lib/i18n/`
- hreflang tags for SEO
- Locale detection with non-blocking banner

### Consent Management

- Privacy-first: blocks all marketing scripts until consent
- `localStorage` for persistence
- Dynamic script loading based on consent state
- Supports GTM, GA4, Meta Pixel, TikTok, Twitter pixels

### Form Handling

- Client-side validation (immediate feedback)
- Server-side validation (security)
- Cloudflare Turnstile for bot protection (preferred)
- Optional reCAPTCHA v3 fallback
- CRM webhook integration

### Performance Optimizations

- Code splitting via Vite manual chunks
- Critical CSS extraction
- Image optimization (WebP/AVIF via Astro Image)
- Lazy loading for below-the-fold content
- Bundle size guardrails (80KB homepage JS budget)

## Component Architecture

### Astro Components

- `HeroCountdown.astro` - Countdown timer with expired state
- `LeadForm.astro` - Multi-step form with validation
- `TestimonialsCarousel.astro` - Slick carousel for testimonials
- `CookieConsent.astro` - Consent banner and manager
- `LanguageSwitcher.astro` - Dropdown for locale selection
- `SeoHead.astro` - SEO meta tags and JSON-LD

### Layout System

- `SiteLayout.astro` - Base layout with header/footer
- Consistent navigation across all pages
- Sticky header for better UX

## Data Flow

### Lead Submission Flow

1. User fills form → Client validation
2. Turnstile verification → Token generation
3. POST to `/api/lead` → Server validation
4. CRM webhook → External system
5. Redirect to `/thank-you` → Success page

### Consent Flow

1. Banner shown on first visit
2. User accepts/rejects/customizes
3. State saved to `localStorage`
4. Marketing scripts loaded/unloaded dynamically
5. Analytics events fired if consented

## Serverless Functions

### Adapter Pattern

- Core handler logic in `functions/handlers/`
- Platform-specific adapters:
  - `functions/vercel/` - Vercel serverless functions
  - `functions/netlify/` - Netlify functions
  - `functions/cloudflare/` - Cloudflare Workers

### API Routes

- `/api/lead` - Lead form submission
- `/api/rum` - Real User Monitoring (optional, disabled in prod)

## Security Considerations

- CSP nonce injection for inline scripts
- Referrer-Policy: strict-origin-when-cross-origin
- Input sanitization and validation
- Rate limiting (via platform)
- Bot protection (Turnstile/reCAPTCHA)

## Deployment Strategy

### Multi-Platform Support

- Single codebase supports all three platforms
- Environment variable `DEPLOY_TARGET` selects adapter
- CI/CD workflows for each platform

### Environment Variables

- Public vars (prefixed with `PUBLIC_`) - Available in client
- Private vars - Server-only (secrets, API keys)

## Testing Strategy

### Unit Tests

- Validation logic
- Utility functions
- Countdown calculations

### E2E Tests

- Form submission flow
- Consent management
- Language switching
- Accessibility checks

### Performance Tests

- Lighthouse CI budgets
- Core Web Vitals monitoring
- Bundle size checks

## Future Enhancements

- Service worker for offline support
- Additional locales (French, German, etc.)
- A/B testing integration
- Advanced analytics dashboards
- Multi-step form wizard











