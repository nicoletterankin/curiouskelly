# The Daily Lesson Marketing Site

A production-ready Jamstack marketing site built with Astro, TypeScript, and Bootstrap. Designed for lead generation with internationalization, consent management, and multi-platform deployment support.

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- Git

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd daily-lesson-marketing

# Install dependencies
npm install

# Copy environment variables
cp .env.example .env

# Start development server
npm run dev
```

Visit `http://localhost:4321` to see the site.

## 📋 Environment Variables

Create a `.env` file in the root directory:

```env
# Locale Configuration
DEFAULT_LOCALE=en-US
OFFER_END_DATE=2025-12-31T23:59:59Z

# Marketing Tags (loaded only after consent)
GTM_ID=
GA4_ID=
META_PIXEL_ID=
TIKTOK_PIXEL_ID=
TWITTER_PIXEL_ID=

# Anti-Bot (Cloudflare Turnstile preferred)
TURNSTILE_SITE_KEY=
TURNSTILE_SECRET_KEY=

# Optional: Google reCAPTCHA v3
RECAPTCHA_SITE_KEY=
RECAPTCHA_SECRET_KEY=

# CRM Integration
CRM_WEBHOOK_URL=http://localhost:3001/api/webhook

# Deployment Target (vercel|netlify|cloudflare)
DEPLOY_TARGET=vercel

# Debug Mode
DEBUG=false

# RUM Endpoint (disabled in prod by default)
ENABLE_RUM=false
```

## 🛠️ Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run lint:fix` - Fix ESLint errors
- `npm run format` - Format code with Prettier
- `npm run format:check` - Check formatting
- `npm run stylelint` - Run Stylelint
- `npm run test` - Run unit tests
- `npm run test:e2e` - Run Playwright E2E tests
- `npm run test:lighthouse` - Run Lighthouse CI
- `npm run mock-crm` - Start mock CRM server

### Running the Mock CRM Server

For local development, start the mock CRM server:

```bash
npm run mock-crm
```

This runs a simple HTTP server on `http://localhost:3001` that logs incoming lead submissions.

## 🌍 Internationalization

### Adding a New Locale

1. Create a new translation file in `src/lib/i18n/` (e.g., `fr-FR.ts`)
2. Add the locale to `src/lib/i18n/index.ts`:
   ```typescript
   export const locales: Locale[] = ['en-US', 'es-ES', 'pt-BR', 'fr-FR'];
   ```
3. Create a localized page directory (e.g., `src/pages/fr-fr/`)
4. Update `sitemap.ts` to include the new locale routes

### Translating Strings

Edit the translation files in `src/lib/i18n/`:
- `en-US.ts` - English (US)
- `es-ES.ts` - Spanish (Spain)
- `pt-BR.ts` - Portuguese (Brazil)

All strings are organized by feature (site, nav, hero, form, footer, consent, thankYou).

## 🚢 Deployment

### Vercel

1. Install Vercel CLI: `npm i -g vercel`
2. Run `vercel` and follow prompts
3. Set environment variables in Vercel dashboard
4. Set `DEPLOY_TARGET=vercel` in environment

### Netlify

1. Install Netlify CLI: `npm i -g netlify-cli`
2. Run `netlify init` and follow prompts
3. Set environment variables in Netlify dashboard
4. Set `DEPLOY_TARGET=netlify` in environment
5. Configure `netlify.toml` if needed

### Cloudflare Pages

1. Install Wrangler CLI: `npm i -g wrangler`
2. Create a Cloudflare Pages project
3. Set environment variables in Cloudflare dashboard
4. Set `DEPLOY_TARGET=cloudflare` in environment
5. Deploy via Git integration or `wrangler pages deploy dist`

### Environment Variables for Production

Ensure all required environment variables are set in your hosting platform:

- `TURNSTILE_SITE_KEY` and `TURNSTILE_SECRET_KEY` (required for form submission)
- `CRM_WEBHOOK_URL` (your CRM endpoint)
- `GTM_ID`, `GA4_ID`, etc. (optional, for analytics)
- `PUBLIC_SITE_URL` (your site URL for SEO)

## 🔒 Consent Management

The site includes a privacy-first consent manager that blocks marketing scripts until users opt-in:

1. **Necessary cookies** - Always enabled (required for site functionality)
2. **Analytics cookies** - Optional (GTM, GA4)
3. **Marketing cookies** - Optional (Meta Pixel, TikTok, etc.)

Users can accept all, reject all, or customize their preferences. Consent state is stored in `localStorage`.

### Adding Marketing Tags

Marketing tags are loaded conditionally via `src/lib/analytics.ts`. To add a new tag:

1. Add environment variable (e.g., `PUBLIC_NEW_PIXEL_ID`)
2. Add loading logic in `loadMarketingPixels()`
3. Ensure it only loads when `consent.marketing === true`

## 📊 Performance

### Performance Budgets

- **LCP**: < 2.5s
- **CLS**: < 0.1
- **INP**: < 200ms
- **JS Bundle**: ≤ 80KB min+gzip (homepage)

### Lighthouse CI

Lighthouse CI runs automatically in GitHub Actions. To run locally:

```bash
npm run test:lighthouse
```

## 🧪 Testing

### Unit Tests

```bash
npm run test
```

Unit tests use Vitest and cover:
- Form validation
- Countdown calculations
- Data sanitization

### E2E Tests

```bash
npm run test:e2e
```

E2E tests use Playwright and cover:
- Lead form submission
- Consent manager
- Language switching
- Accessibility

### Running Tests in CI

All tests run automatically on push/PR via GitHub Actions. See `.github/workflows/build-test.yml`.

## 📁 Project Structure

```
daily-lesson-marketing/
├── src/
│   ├── components/       # Astro components
│   ├── layouts/          # Page layouts
│   ├── lib/              # Utilities and helpers
│   │   ├── i18n/         # Translation dictionaries
│   │   ├── analytics.ts   # Analytics integration
│   │   ├── consent.ts    # Consent manager
│   │   ├── countdown.ts  # Countdown logic
│   │   ├── countries.ts  # Country/region data
│   │   └── validation.ts # Form validation
│   ├── pages/            # Astro pages (routes)
│   │   ├── api/          # API routes
│   │   ├── es-es/        # Spanish pages
│   │   └── pt-br/        # Portuguese pages
│   └── styles/            # SCSS styles
├── functions/             # Serverless function adapters
├── public/                # Static assets
├── tests/                 # Test files
│   ├── e2e/              # Playwright tests
│   ├── lighthouse/       # Lighthouse CI config
│   └── unit/             # Vitest tests
└── scripts/               # Build scripts
```

## 🔧 Configuration

### Astro Config

Edit `astro.config.mjs` to:
- Change adapter (Vercel/Netlify/Cloudflare)
- Configure build optimizations
- Set image domains

### TypeScript

TypeScript config is in `tsconfig.json`. Path aliases:
- `@/*` → `src/*`
- `@components/*` → `src/components/*`
- `@lib/*` → `src/lib/*`

### Linting

- **ESLint**: `.eslintrc.cjs`
- **Prettier**: `.prettierrc`
- **Stylelint**: `stylelint.config.cjs`

## 🎨 Styling

Styles use Bootstrap 4.6 via SCSS. Custom overrides in `src/styles/main.scss`.

### Critical CSS

Critical CSS is extracted automatically by Astro. Ensure above-the-fold styles are optimized.

## 📱 PWA Support

The site includes a basic PWA setup:
- `public/manifest.json` - App manifest
- Service worker (via Workbox, can be added)

## 🔍 SEO

- Automatic sitemap generation (`/sitemap.xml`)
- `robots.txt` configuration
- hreflang tags for i18n
- Schema.org JSON-LD markup
- Open Graph and Twitter Card meta tags

## 🐛 Troubleshooting

### Build Errors

- Ensure Node.js 18+ is installed
- Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`
- Check environment variables are set correctly

### Form Submission Fails

- Verify Turnstile keys are set correctly
- Check CRM webhook URL is accessible
- Review browser console for errors
- Ensure CORS is configured on CRM endpoint

### Analytics Not Loading

- Verify consent manager is working
- Check GTM/GA4 IDs are set
- Ensure marketing consent is granted
- Check browser console for script errors

## 📚 Additional Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture and design decisions
- [PERFORMANCE.md](./PERFORMANCE.md) - Performance optimization guide
- [PRIVACY.md](./PRIVACY.md) - Privacy and data handling documentation
- [RUNBOOK.md](./RUNBOOK.md) - Operations and deployment runbook

## 🤝 Contributing

1. Create a feature branch
2. Make changes
3. Run tests: `npm run test && npm run test:e2e`
4. Submit a pull request

## 📄 License

This project uses neutral branding ("The Daily Lesson") and placeholder assets. No proprietary content is included.

## 🙏 Acknowledgments

Built with:
- [Astro](https://astro.build/) - Static site framework
- [Bootstrap](https://getbootstrap.com/) - CSS framework
- [Playwright](https://playwright.dev/) - E2E testing
- [Vitest](https://vitest.dev/) - Unit testing
- [Lighthouse CI](https://github.com/GoogleChrome/lighthouse-ci) - Performance testing


















