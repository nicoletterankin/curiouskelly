# Lesson of the Day Web Presence Map (Draft)

> Domains: `lessonoftheday.com`, `reinmaker.com`, `curiouskelly.com`, `thedailylesson.com`, `ilearnhow.com`

## Goals
- Present a cohesive brand while targeting distinct customer segments (game, professional, personal teacher, hardware).
- Cross-link every property back to Lesson of the Day PBC and provide clear upgrade/upsell pathways.
- Maintain consistent analytics, SEO structure, accessibility, and localization.

## Domain Roles

| Domain | Primary Audience | Purpose | Key CTAs |
|--------|------------------|---------|----------|
| `lessonoftheday.com` | Investors, press, partners | Parent hub, mission, roadmap, links to all products | “Explore Products”, “Request Demo”, newsletter signup |
| `reinmaker.com` | Learners & parents, game studios | Gamified learning experience, leaderboards, trailers | “Play Reinmaker”, “Family Plans”, “Download App” |
| `curiouskelly.com` | Families, daily learners | Personal teacher, micro-sessions, daily streaks | “Start Free Trial”, “Meet Kelly”, “Download App” |
| `thedailylesson.com` | Schools, enterprises | Professional edition with compliance, analytics | “Book Enterprise Demo”, “Download Brochure”, “Contact Sales” |
| `ilearnhow.com` | Hardware buyers (future) | Smart device ecosystem (voice, sensors) | “Join Waitlist”, “Hardware Specs”, “Developer Program” |

## Cross-Linking Rules
- Every footer includes links to all five domains plus privacy, terms, and support pages.
- Each product site highlights the other offerings in a “More from Lesson of the Day” section (cards with CTA).
- Global navigation on `lessonoftheday.com` surfaces product landing pages and investor resources.
- Use consistent UTM parameters (`utm_source=lotd-hub`, etc.) when cross-linking for analytics.

## Analytics & Tagging
- Shared Google Tag Manager container with product-specific data layers (`product: reinmaker | curious_kelly | daily_lesson | ilearnhow`).
- Segment or Mixpanel workspace with unified user IDs where cross-domain login is supported.
- Cookie consent banner aligned with GDPR/CPRA; allow per-domain opt-in storage.

## Localization & Accessibility
- Default locales: EN, ES, FR. Provide locale switcher in header; remember preference via cookie.
- Ensure WCAG 2.1 AA compliance (contrast ratios, keyboard nav, aria labels).
- Provide region-specific content (pricing, testimonials) pulled from CMS using locale and region flags.

## Content Ownership
- Central CMS (e.g., Contentful/Sanity) hosts shared copy modules (mission, FAQ, brand story).  
- Product marketing teams own hero sections, feature grids, testimonials specific to their audience.  
- Legal/Compliance manages privacy, terms, accessibility pages globally.

## Deployment & Hosting
- Use Vercel/Cloudflare Pages with separate projects per domain but shared component library.  
- CI workflow enforces lighthouse checks (performance, accessibility, SEO).  
- Blue/green deploys for major marketing campaigns; maintain rollback ability.

## Next Steps
1. Create wireframes per domain and validate messaging hierarchy.  
2. Establish shared design system tokens (colors, typography, iconography).  
3. Implement analytics instrumentation plan (GTM tags, consent management).  
4. Draft copy for landing pages and cross-linking sections.  
5. Coordinate with legal for domain-specific disclosures (COPPA for Reinmaker, FERPA for Daily Lesson).













