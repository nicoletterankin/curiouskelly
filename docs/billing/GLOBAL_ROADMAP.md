# Global Billing & Entitlement Roadmap (Draft)

> Owner: Billing Platform Team (Lesson of the Day PBC)  
> Last updated: 2025-10-30

## Objectives
- Support every Lesson of the Day product (Reinmaker, The Daily Lesson, Curious Kelly, future iLearnHow hardware) with a unified billing core.
- Offer region-appropriate payment methods and pricing, while sharing entitlements across products.
- Keep compliance (PCI, SOC2, GDPR, COPPA, FERPA) intact as we scale globally.

## Platform Responsibilities
- Tokenize and securely store payment instruments (Stripe Radar + Vault service).  
- Process payments across channels (in-app, web, hardware kiosks).  
- Manage subscriptions, refunds, proration, trial logic, and dunning.  
- Issue signed entitlements consumed by product backends via `/entitlements/{userId}` API.  
- Maintain tax/VAT handling and localized invoicing.  
- Expose reporting dashboards for Finance and CS operations.

## Product Consumers
- **Reinmaker (Game)** – uses entitlements `reinmaker.game_pass`, `reinmaker.family_pack`.  
- **The Daily Lesson (Professional)** – entitlements `daily_lesson.standard`, `daily_lesson.enterprise`.  
- **Curious Kelly (Personal Teacher)** – entitlements `curious_kelly.personal`, `curious_kelly.family`.  
- **iLearnHow (Hardware)** – future entitlements `ilearnhow.device_bundle`, `ilearnhow.subscription`.

## Phased Rollout

### Phase 1 – United States & Canada (Q4 2025)
- **Web**: Stripe (cards, ACH, Apple Pay, Google Pay).  
- **Mobile**: Apple IAP, Google Play Billing.  
- **Compliance**: PCI DSS SAQ A, state tax nexus (Avalara).  
- **Deliverables**:
  - Billing service MVP (`billing-service/` repo) with subscription APIs.
  - Unified entitlement issuance + webhook fan-out to product backends.
  - Customer portal for plan management (Stripe Billing portal skin).

### Phase 2 – European Union & UK (Q1 2026)
- **Payments**: SEPA Direct Debit, SOFORT, Bancontact, iDEAL, PayPal.  
- **Tax**: VAT MOSS registration; digital services compliance.  
- **Currency**: EUR, GBP; multi-currency pricing matrix.  
- **Localization**: Support EU privacy consents (GDPR).  
- **Deliverables**:
  - Localized checkout flows with PSD2 SCA support.  
- Update invoicing to include VAT IDs and reverse-charge logic.

### Phase 3 – APAC & LatAm (Q2–Q3 2026)
- **Payments**: UPI, Paytm, Alipay, WeChat Pay, GrabPay, PIX, Boleto, MercadoPago.  
- **Carrier billing**: Evaluate Boku/Fortumo for mobile-first regions.  
- **Currency**: INR, CNY, SGD, BRL, MXN.  
- **Compliance**: Local data residency where required (e.g., India).  
- **Deliverables**:
  - Regional pricing catalogs; currency hedging rules.  
  - Support tax withholding certificates (GST, ISS) as applicable.

### Phase 4 – Enterprise & Institutional (Parallel track)
- **Invoicing**: Manual procurement, PO-based billing, net 30/60 terms.  
- **Integrations**: Workday, SAP Ariba, LMS connectors.  
- **Contracts**: Master Service Agreements, SLAs, compliance attestations.  
- **Deliverables**:
  - Dedicated enterprise billing workflow with approval chains.  
  - API for automated seat provisioning (`POST /enterprise/licenses`).

## Architecture Sketch (Draft)
```
[Clients] → [Checkout UI] → [Billing Service (LOT-D)] → [Payment Processors]
                                 ↓                        ↓
                          [Entitlement Engine]      [Tax Engine]
                                 ↓
                      [Products consume via API]
```

## Entitlement Contract (Draft JSON)
```json
{
  "userId": "uuid",
  "product": "reinmaker",
  "plan": "game_pass",
  "status": "active",
  "issuedAt": "2025-10-30T15:04:05Z",
  "expiresAt": "2025-11-30T15:04:05Z",
  "features": ["multiplayer", "daily_streak_boost"],
  "source": "stripe",
  "region": "US"
}
```

## Integration Checklist
- [ ] Payment processor evaluation (fees, coverage, compliance).  
- [ ] Secrets management (Doppler/AWS Secrets Manager).  
- [ ] Fraud & risk mitigation (Stripe Radar + internal heuristics).  
- [ ] Refund policy & customer support workflows.  
- [ ] Legal review for Terms of Service & privacy updates per region.  
- [ ] QA scenarios covering cancellations, proration, charge disputes.  
- [ ] Monitoring dashboards (failed payments, churn, MRR, LTV).

## Risks & Mitigations
- **High processor fees** → Negotiate volume discounts; route payments by region.  
- **Complex tax rules** → Use Avalara/TaxJar; keep rules in config, not code.  
- **Entitlement drift between systems** → Nightly reconciliation job + webhooks.  
- **Chargeback/fraud spikes** → Enforce 3DS/SCA where mandated; monitor velocity.  
- **Regulatory changes** → Maintain legal calendar; subscribe to compliance updates.

## Next Actions
1. Draft technical design for billing service (ERDs, API spec).  
2. Stand up staging Stripe account + sandbox Play/App Store configs.  
3. Define pricing catalogs and localized terms per region.  
4. Align finance, legal, support teams on rollout schedule.  
5. Update CLAUDE.md once billing APIs are published.













