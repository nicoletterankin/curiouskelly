# Privacy & Data Handling

## Data Collection

### Information Collected

- **Name** (first and last)
- **Email address**
- **Phone number** (international format)
- **Country and region**
- **Marketing preferences** (opt-in checkbox)

### How Data is Used

- **Lead generation**: Data forwarded to CRM system
- **Communications**: Email/phone for follow-up (if consented)
- **Analytics**: Aggregated usage data (if consented)
- **Marketing**: Targeted advertising (if consented)

## Consent Management

### Cookie Categories

1. **Necessary** (always enabled)
   - Required for site functionality
   - Cannot be disabled

2. **Analytics** (optional)
   - Google Tag Manager
   - Google Analytics 4
   - Usage analytics

3. **Marketing** (optional)
   - Meta Pixel
   - TikTok Pixel
   - Twitter Pixel
   - Other advertising pixels

### Consent Storage

- Consent state stored in `localStorage`
- Key: `consentState`
- Never expires (user can change anytime)

## Data Security

### Transmission

- All form submissions use HTTPS
- Data encrypted in transit

### Storage

- Serverless functions process data
- No database storage (stateless)
- Data forwarded to CRM immediately
- No persistent storage on our servers

### Access Control

- Serverless functions run in secure environment
- Environment variables for secrets
- No hardcoded credentials

## Third-Party Services

### CRM Integration

- Data forwarded to configured `CRM_WEBHOOK_URL`
- Payload includes all form fields + metadata
- CRM system responsible for data storage/retention

### Analytics Providers

- Google Tag Manager (if consented)
- Google Analytics 4 (if consented)
- Respects Do Not Track headers

### Bot Protection

- Cloudflare Turnstile (preferred)
- Optional: Google reCAPTCHA v3
- No personal data collected by verification

## User Rights

### Access

Users can request access to their data via email to privacy@thedailylesson.com

### Deletion

Users can request deletion via email to privacy@thedailylesson.com

### Opt-Out

- Marketing emails: Unsubscribe link in emails
- Analytics: Adjust consent preferences
- Cookies: Use browser settings or consent manager

## Privacy Policy

Full privacy policy available at `/privacy/`.

## Compliance

### GDPR

- Consent required for marketing/analytics
- Right to access/deletion
- Data portability (via CRM export)

### CCPA

- Clear opt-out mechanisms
- No sale of personal information
- Transparent data collection

## Contact

Privacy questions: privacy@thedailylesson.com











