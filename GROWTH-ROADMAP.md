# Curious Kelly - Growth Roadmap
Generated: 2025-12-15

## Current State
- **Site:** WORKING (Tier 1 checks: `/`, `/learn.html?day=17`, loader, Day 17 pack all HTTP 200)
- **Discoverability:** GOOD foundation in place (robots + sitemaps deployed)
- **User Experience:** GOOD core loop (lesson loads, audio/video playback starts), but needs polish audits (first-time + mobile)

## Priority 1: Immediate (This Week)
- [ ] **Verify Google Search Console** (ownership + submit both sitemaps) — see `YOUR_ACTION_REQUIRED.md`
- [ ] **Run an incognito first-time walkthrough** and record friction points (homepage clarity, first click to lesson, any surprises like paywall dialogs)
- [ ] **Reduce noisy production console logging** (keep errors; downgrade debug-only warnings behind explicit `debug=true`)

## Priority 2: Short-term (Next 2 Weeks)
- [ ] **Improve SERP + share previews**: confirm OG image paths exist and render correctly on X/Slack previews
- [ ] **Lesson entry reliability**: validate autoplay/play UX across mobile Safari + Chrome (clear affordance if autoplay blocked)
- [ ] **Index coverage**: confirm Google starts discovering `/learn.html?day=N` URLs via `sitemap-lessons.xml` and fix any “Crawled - currently not indexed” patterns

## Priority 3: Medium-term (Next Month)
- [ ] **Analytics instrumentation**: track landing → start lesson → completion funnel; measure drop-offs and fix highest-impact UX steps
- [ ] **Performance budgets**: keep sub-second nav feel; tighten caching policies only where safe
- [ ] **On-site SEO**: add/iterate copy and structured data based on Search Console queries (no keyword stuffing)

## Priority 4: Long-term (Q1 2026)
- [ ] Mobile app packaging + store readiness
- [ ] Subscription/billing iteration (based on funnel data)
- [ ] Partnerships (schools/orgs) once conversion story is proven

## Metrics to Track (weekly)
- Unique visitors (home)
- Click-through to learn (`/learn.html`)
- Lesson start rate
- Lesson completion rate
- Return visitor rate (next day)
- Search Console: impressions, clicks, CTR, indexed pages



