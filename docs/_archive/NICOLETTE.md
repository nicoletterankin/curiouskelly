# NICOLETTE.md — Human-Only Launch Checklist
> Every item below still needs human action. The assistant will handle everything else.

**Status**: Live working list (update as you complete items)  
**Last updated**: 2025-10-30

---

## ✅ Already handled by the assistant
- Drafting store listing copy, landing page copy, and launch announcement content (you’ll just review/approve in later PRs).
- Technical implementation, automation scripts, and asset manifests referenced in `CLAUDE.md`.

Everything that follows requires human accounts, approvals, payments, signatures, or business decisions that only you can make. Each task contains the information and prerequisites so you can execute without hunting for context.

---

## 🏢 Account Registrations & Financial Setup

### 1. Apple Developer Program (Company enrollment – $99/year)
- [x] **Enroll and activate the Team Agent account** *(Order # W1619886043 processed Nov 1, 2025)*
  - Membership payment confirmed via Apple Store receipt email. Next human step: complete App Store Connect tax/banking once EIN + routing info are ready.
  - Reference checklist: `Curious-Kellly_GTM_Checklists.md` → Apple App Store.

### 2. Google Play Console (Developer account – $25 one-time)
- [ ] **Register and verify**
  - Visit `https://play.google.com/console/signup` with a Google account dedicated to Lesson of the Day.
  - Pay the $25 registration fee (credit card, no refunds).
  - Complete developer profile (legal business name, address, contact, support URL, privacy policy URL placeholder).
  - Finish identity verification (government-issued ID and possible business registration proof). This can take 48–72 hours.
  - Set up the payments profile for payouts (bank account, tax details). Reference: `Curious-Kellly_GTM_Checklists.md` → Google Play.

### 3. Stripe Account (Primary payment processor)
- [ ] **Create and verify Stripe account**
  - Sign up at `https://dashboard.stripe.com/register`.
  - Provide business details: legal entity, EIN, personal info for beneficial owners (>25%), business address, customer support phone/email, statement descriptor.
  - Link a business bank account (routing + account numbers).
  - Activate Radar (fraud prevention) and enable webhook endpoints (these will be supplied by the billing service once ready).
  - Set up monitoring alerts (failed payments, disputes) routed to `finance@lessonoftheday.com` (or preferred alias).
  - Reference: `docs/billing/GLOBAL_ROADMAP.md` → Phase 1.

### 4. Secondary payment processors (Phase 2+ rollout)
- [ ] **PayPal Business** – register at `https://www.paypal.com/bizsignup/` (need EIN, bank account, responsible party SSN); enable reference transactions if subscriptions will be used.
- [ ] **Regional wallets** – queue applications for UPI/Paytm (India), Alipay/WeChat Pay (China), PIX/MercadoPago (LatAm). Each requires local business registration or payment partner—note lead time (2–6 weeks).
- [ ] **Carrier billing vendors** – shortlist Boku/Fortumo, begin vendor onboarding (NDA + compliance questionnaires).

### 5. Tax compliance platform
- [ ] **Select and onboard Avalara or TaxJar**
  - Create account, add company profile, import product catalog for digital goods, configure jurisdictions (US states + EU countries).
  - Complete Nexus questionnaire to identify where sales tax/VAT must be collected.
  - Ensure access for finance + engineering (API keys for billing service).

### 6. Domain portfolio & DNS
- [ ] **Register/transfer domains**: `lessonoftheday.com`, `reinmaker.com`, `curiouskelly.com`, `thedailylesson.com`, `ilearnhow.com` (if not already owned).
  - Recommended registrar: Cloudflare Registrar or Namecheap. Use the same account for all domains.
- [ ] **Configure DNS + SSL**
  - Point root and `www` records to planned hosting (Cloudflare Pages/Vercel). Set mandatory CAA entries for issuing CA.
  - Enable automatic SSL (Let’s Encrypt or Cloudflare Universal SSL) and HSTS.
- [ ] **Create domain ownership proofs** (TXT records) for Apple, Google Play, and email service.

### 7. Hosting accounts
- [ ] **Vercel or Cloudflare Pages** – create projects for each domain and connect Git repos once marketing sites are ready.
- [ ] **Email & support tooling** – set up Google Workspace or equivalent with shared inboxes (`support@`, `legal@`, `billing@`).

### 8. Third-party API accounts
- [x] **OpenAI** – $50 credit added Nov 1, 2025 (Tier 3 limits email confirmed). Still need to set soft/hard alerts in usage dashboard.
- [ ] **ElevenLabs** – verify Kelly/Kyle voice access, upgrade to plan covering anticipated runtime (≥60 minutes training data must already be stored). Record API keys in secrets manager.
- [ ] **Analytics (Mixpanel/Amplitude)** – create workspace, define project (Curious Kelly), configure retention dashboard templates, and invite product/data team emails.

---

## 📱 Store Submission Requirements (Human-only portions)

### Apple App Store Connect
- [ ] **App Privacy nutrition label**
  - Sign into App Store Connect → App → App Privacy. Use the data inventory prepared in `CK_Launch-Checklist.csv` (LC-002) to answer tracking and data collection questions.
  - Decide on tracking disclosures (default should be “No tracking”) and specify data collected per feature.
- [ ] **Screenshots & preview video capture**
  - Capture on physical devices or TestFlight builds: 6.7" (iPhone Pro Max) and 6.1" (iPhone Pro). Use App Store guidelines (no status bar overlays, use Light/Dark appropriate).
  - Produce 15–30 second preview video demonstrating Kelly voice/avatar, include accessibility captions.
- [ ] **Finalize store copy**
  - Assistant will supply draft short/full descriptions and keywords; you confirm wording and approvals before submission.
- [ ] **Age rating questionnaire + export compliance**
  - Complete `General Information` → `Age Rating`; answer “No” to restricted content unless features dictate otherwise. For encryption/export, declare use of standard TLS and voice chat (select appropriate exemption).
- [ ] **Create In-App Purchase products**
  - In App Store Connect → Features → In-App Purchases: set up Monthly, Annual, Family subscriptions with localized pricing, review screenshots, subscription group, free trial logic.
  - Attach server-to-server notification endpoint (from billing service) when available.
- [ ] **Review notes**
  - Prepare 3–5 sentences summarizing AI voice interactions, microphone usage consent, and link to privacy policy. Provide demo credentials if login required.

### Google Play Console
- [ ] **Data Safety form**
  - Under App Content → Data Safety, mirror the same data inventory as Apple. Specify purposes, data sharing, and encryption.
- [ ] **Account deletion URL**
  - Ensure `https://lessonoftheday.com/account-delete` (or equivalent) exists and is functional before submitting the form.
- [ ] **IARC rating questionnaire**
  - Complete the online questionnaire via Play Console; export the certificate and store in `compliance/` folder.
- [ ] **Store listing approvals**
  - Upload assistant-provided copy, screenshots (phone + tablet), 1024×500 feature graphic, and promo video (YouTube).
- [ ] **Play Billing products**
  - Create base plan + offer for monthly/annual/family subscriptions. Link to test users for licensing, and populate tax categories (Digital content → Apps).

### GPT Store / Claude Artifacts
- [ ] **OpenAI Builder Profile verification** – follow prompts at `https://platform.openai.com/assistants/` (requires active Plus/Pro/Enterprise account). Provide public-facing display name, logo, and description.
- [ ] **Final review of GPT listing** – assistant will draft listing copy and demo script; you confirm compliance and toggle visibility to “Everyone.”

---

## ⚖️ Legal, Compliance & Policies

- [ ] **Terms of Service**
  - Draft (assistant can supply template) covering: service description, acceptable use, billing terms, cancellation/refund policy, data ownership, arbitration venue.
  - Ensure alignment with Apple/Google platform terms and COPPA/FERPA obligations.
  - Engage legal counsel for review and sign-off.
- [ ] **Privacy Policy**
  - Include data categories, purpose, retention, third-party sharing, cookies, user rights (GDPR/CPRA), contact for data requests, children’s privacy statement.
  - Publish at `https://lessonoftheday.com/privacy` before store submissions.
- [ ] **Legal review** – obtain written sign-off (email or PDF) from counsel; archive in `compliance/legal-approvals/`.
- [ ] **FERPA compliance package** (for Daily Lesson)
  - Document student data flows, create Data Protection Agreement template, and add contact for district agreements.
- [ ] **Accessibility statement**
  - Publish WCAG 2.1 AA statement describing accessibility commitments and contact path (`accessibility@lessonoftheday.com`).
- [ ] **Enterprise compliance roadmap**
  - Kick off SOC2 readiness (choose auditor), outline timeline, and create DPA template for enterprise customers.

---

## 💰 Business & Finance Approvals

- [ ] **Pricing catalogs**
  - Approve target price points per product/region (USD baseline, convert for EU/UK/EUROPE & APAC). Document decisions in `pricing/pricing-catalog.xlsx`.
- [ ] **Currency hedging strategy**
  - Decide on FX buffer or hedging (consult finance advisor). Set policy for quarterly price review.
- [ ] **Monthly recurring cost approvals**
  - Sign off on projected spend: OpenAI, ElevenLabs, Stripe/processor fees, CDN, hosting, analytics. Capture in finance forecast.
- [ ] **One-time launch costs**
  - Approve developer program fees, domain acquisitions, legal retainers. Log in `finance/launch-costs.xlsx`.

---

## 🎨 Creative & Content Sign-Offs (Assistant supplies drafts)

- [ ] **Approve landing page copy & designs** – review assistant-provided copyblocks/wireframes for each domain, then greenlight for implementation.
- [ ] **Approve brand guidelines** – confirm color palette, typography, iconography, logo usage produced by assistant/design tooling.
- [ ] **Approve launch announcements** – finalize press release, email, and social posts drafted by assistant.
- [ ] **Lesson content QA** – review the first 10 lessons (EN+ES/FR) for tone accuracy; sign off before scaling to 90.
- [ ] **Voice persona sign-off** – listen to the latest Kelly/Kyle audio proofs (≥60 min each) and confirm personas match brand expectations.

---

## 👥 Beta & Support Operations

- [ ] **Recruit TestFlight cohort (300)**
  - Prepare invite list (CSV with emails), send via App Store Connect → TestFlight → External Testers. Include NDAs if required.
- [ ] **Recruit Google Play internal testers (300)**
  - Collect Gmail addresses, add to “Internal Testing” track, distribute instructions.
- [ ] **Feedback triage plan**
  - Set up tracking sheet or Linear/Jira project for beta feedback. Assign owners for crash logs, UX feedback, feature requests.
- [ ] **Support channel provisioning**
  - Configure `support@lessonoftheday.com`, choose helpdesk tool (Zendesk/HelpScout/Freshdesk), set SLAs (24h initial response, P1 = 2h), create canned responses.
- [ ] **Support playbook**
  - Document refund policy, bug escalation path, emergency contacts, hours of operation. Store in `operations/support-runbook.md`.

---

## 🚀 Launch Operations

- [ ] **Final store submissions**
  - After QA sign-off, in App Store Connect select build, add release notes, click “Submit for Review.”
  - In Google Play Console, promote the release from Internal testing to Production, confirm target SDK 35, complete declarations, click “Submit for review.”
- [ ] **Launch-day monitoring**
  - Schedule coverage for day 0 & 1 to watch dashboards (voice latency, crash-free rate, payment success). Ensure PagerDuty/on-call rotation exists.
- [ ] **Announcements**
  - Publish coordinated posts (press release, social, email) at agreed launch time. Monitor responses and engage.
- [ ] **Post-launch review**
  - Check store reviews twice daily, respond within 24h. Compile Day-1 metrics (activation, retention, revenue) for review meeting.

---

## 🔐 Secrets & Security

- [ ] **Secrets manager rollout**
  - Choose Doppler or AWS Secrets Manager, create project, import existing secrets (OpenAI, ElevenLabs, Stripe, etc.).
  - Rotate keys before production, enforce access controls, and update deployment pipelines to read from secrets manager.
- [ ] **Security contacts**
  - Establish `security@lessonoftheday.com` alias and incident response doc.

---

## 📋 Store Platform Actions (Manual)

- [ ] **App Store Connect operational tasks**
  - Approve each TestFlight build for external testers.
  - Upload production builds via Xcode/Transporter or approve CI/CD uploads.
  - Respond to App Review or compliance questions within 24h.
- [ ] **Google Play Console operational tasks**
  - Upload signed AAB, configure staged rollout (if using), and respond to policy review emails quickly.

---

## 🎯 Strategic Decisions

- [ ] **Finalize launch sequencing** – confirm which SKU launches first (Curious Kelly mobile, Reinmaker game, Daily Lesson enterprise) and communicate to engineering.
- [ ] **Feature prioritization** – decide what can slip if timelines compress; update roadmap accordingly.
- [ ] **Lesson calendar approval** – sign off on first 30-day curriculum + age personas.

---

## 📞 Integration Coordination

- [ ] **Reinmaker API production details** – once backend is live, supply final base URL and OAuth client credentials so assistant can update `docs/reinmaker/API_OVERVIEW.md`.
- [ ] **Billing architecture approval** – review `docs/billing/GLOBAL_ROADMAP.md` technical design, confirm processor mix and regions before implementation begins.
- [ ] **Web analytics plan approval** – approve GTM/data layer schema before deployment (`docs/web/SITE_MAP.md` references instrumentation requirements).

---

## 🚦 Critical Path (deadline reminders)

- **Before Week 3**: Fund OpenAI account *(done Nov 1)*, activate ElevenLabs voices.
- **Before Week 7**: Apple Developer enrollment, Google Play Console, Stripe verification.
- **Before Week 11**: Terms of Service + Privacy Policy finalized; beta tester lists ready.
- **Before Week 12**: Store assets uploaded, store submissions sent, support channels staffed.

---

Keep this file updated as you complete each item. The assistant will continue preparing technical assets, drafts, and automation so you can focus on the human-only actions above.

