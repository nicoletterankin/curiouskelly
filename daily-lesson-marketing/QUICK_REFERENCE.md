# 🚀 Quick Reference Cheat Sheet

## File Structure Quick Guide

```
src/
├── pages/          → URLs (file = route)
│   ├── index.astro → /
│   ├── about.astro → /about
│   └── api/        → API endpoints
├── components/     → Reusable UI pieces
├── layouts/        → Page wrappers
└── lib/            → Utilities & helpers
```

## Common Commands

```bash
npm run dev        # Start dev server (localhost:4321)
npm run build      # Build for production
npm run preview    # Preview production build
npm run test       # Run unit tests
npm run test:e2e   # Run E2E tests
npm run lint       # Check code quality
```

## Astro Component Template

```astro
---
// FRONTMATTER: Server-side (build time)
import Component from './Component.astro';

const data = 'Hello';
---

<!-- TEMPLATE: HTML -->
<h1>{data}</h1>
<Component />

<script>
  // CLIENT: Browser-side JavaScript
  console.log('Loaded!');
</script>

<style>
  /* SCOPED CSS */
  h1 { color: blue; }
</style>
```

## Getting Translations

```astro
---
import { getTranslations } from '@lib/i18n';
const t = getTranslations('en-US'); // or 'es-ES', 'pt-BR'
---

<h1>{t.hero.headline}</h1>
```

## Form Submission Pattern

```javascript
// 1. Validate
const validation = validateLeadForm(data, t);
if (!validation.valid) return;

// 2. Submit
const response = await fetch('/api/lead', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data)
});

// 3. Handle response
if (response.ok) {
  window.location.href = '/thank-you';
}
```

## Environment Variables

```env
# Required
TURNSTILE_SITE_KEY=...
TURNSTILE_SECRET_KEY=...
CRM_WEBHOOK_URL=...

# Optional
GTM_ID=...
GA4_ID=...
PUBLIC_SITE_URL=https://...
```

## Adding a New Page

1. Create `src/pages/new-page.astro`
2. Use `SiteLayout` component
3. Add content
4. Visit `/new-page`

## Adding a New Locale

1. Create `src/lib/i18n/xx-XX.ts` (translation file)
2. Update `src/lib/i18n/index.ts` (add to locales array)
3. Create `src/pages/xx-xx/index.astro` (localized page)

## Consent Management

```typescript
// Get consent state
const consent = consentManager.get();

// Update consent
consentManager.set({ analytics: true });

// Listen for changes
consentManager.onChange((state) => {
  if (state.analytics) loadGTM();
});
```

## API Route Pattern

```typescript
// src/pages/api/endpoint.ts
import type { APIRoute } from 'astro';

export const POST: APIRoute = async ({ request }) => {
  const body = await request.json();
  
  // Process...
  
  return new Response(
    JSON.stringify({ success: true }),
    { status: 200, headers: { 'Content-Type': 'application/json' } }
  );
};
```

## Testing Patterns

**Unit Test**:
```typescript
test('validates email', () => {
  expect(validateEmail('test@example.com')).toBe(true);
});
```

**E2E Test**:
```typescript
test('submits form', async ({ page }) => {
  await page.goto('/');
  await page.fill('#email', 'test@example.com');
  await page.click('#submit-btn');
});
```

## Deployment Checklist

- [ ] Environment variables set
- [ ] Tests passing
- [ ] Build succeeds
- [ ] Preview looks good
- [ ] Form submission works
- [ ] i18n working
- [ ] Consent manager working

## Debugging Tips

1. **Check build**: `npm run build` → inspect `dist/`
2. **Preview**: `npm run preview` → see production build
3. **Console**: Browser DevTools → Console tab
4. **Network**: DevTools → Network tab → check API calls
5. **Mock CRM**: `npm run mock-crm` → test form locally

## Common Patterns

**Conditional Rendering**:
```astro
{condition && <div>Content</div>}
```

**Looping**:
```astro
{items.map(item => <div>{item.name}</div>)}
```

**Dynamic Classes**:
```astro
<div class={`base ${isActive ? 'active' : ''}`}>
```

**Client Hydration**:
```astro
<InteractiveComponent client:load />
```

## File Extensions Explained

- `.astro` → Astro component (HTML + frontmatter)
- `.ts` → TypeScript file
- `.tsx` → React component (TypeScript)
- `.scss` → SCSS stylesheet

## Key Imports

```typescript
// Translations
import { getTranslations } from '@lib/i18n';

// Validation
import { validateLeadForm } from '@lib/validation';

// Consent
import { consentManager } from '@lib/consent';

// Analytics
import { analytics } from '@lib/analytics';

// Countries
import { countries } from '@lib/countries';
```

## Git Workflow

```bash
git checkout -b feature/new-feature
# Make changes
npm run test
npm run lint
git commit -m "Add new feature"
git push origin feature/new-feature
# Create PR
```

## Performance Targets

- LCP: < 2.5s
- CLS: < 0.1
- INP: < 200ms
- JS Bundle: ≤ 80KB gzipped

## Platform-Specific Notes

**Vercel**:
- Auto-detects Astro
- Set `DEPLOY_TARGET=vercel`

**Netlify**:
- Needs `netlify.toml`
- Set `DEPLOY_TARGET=netlify`

**Cloudflare**:
- Needs `wrangler.toml` (optional)
- Set `DEPLOY_TARGET=cloudflare`

## Component Props Pattern

```astro
---
interface Props {
  title: string;
  optional?: boolean;
}

const { title, optional = false } = Astro.props;
---

<h1>{title}</h1>
{optional && <p>Optional content</p>}
```

## Environment Variable Access

**Server-side** (frontmatter):
```astro
---
const apiKey = import.meta.env.API_KEY;
---
```

**Client-side** (script):
```javascript
const siteUrl = import.meta.env.PUBLIC_SITE_URL;
```

**Note**: Only `PUBLIC_*` vars available in client!

## Common Errors & Fixes

**"Cannot find module"**:
→ Check import path
→ Run `npm install`

**"Type error"**:
→ Check TypeScript types
→ Run `npm run build` to see errors

**"Form not submitting"**:
→ Check Turnstile keys
→ Check CRM webhook URL
→ Check browser console

**"Translations not working"**:
→ Check locale code matches
→ Check translation file exists
→ Check `getTranslations()` call

## Useful Links

- Astro Docs: https://docs.astro.build/
- TypeScript: https://www.typescriptlang.org/
- Bootstrap: https://getbootstrap.com/docs/4.6/


















