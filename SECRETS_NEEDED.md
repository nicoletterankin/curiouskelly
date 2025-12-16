# Secrets & Keys Needed for Production

To activate the backend and Stripe payments, please gather the following keys. You will need to provide these to your hosting provider (Railway) as Environment Variables.

## 1. Core Configuration
- `NODE_ENV`: `production`
- `FRONTEND_URL`: `https://curiouskelly.com` (or your Vercel/Netlify URL)
- `JWT_SECRET`: A long random string (e.g. generated via `openssl rand -base64 32`)

## 2. Database (PostgreSQL)
- `DATABASE_URL`: Your Postgres connection string (Railway provides this automatically when you add a PG plugin).

## 3. Stripe Payments
You need a Stripe account. Create these products in your Stripe Dashboard and get their IDs.
- `STRIPE_SECRET_KEY`: `sk_live_...`
- `STRIPE_PUBLISHABLE_KEY`: `pk_live_...`
- `STRIPE_WEBHOOK_SECRET`: `whsec_...` (From the Webhooks section after pointing it to `your-backend.railway.app/webhook`)

**Product Price IDs:**
- `PRICE_ID_PERSONAL`: ID for "Scholar" plan (e.g. `price_123xyz...`)
- `PRICE_ID_FAMILY`: ID for "Family" plan (optional for now)
- `PRICE_ID_GIFT`: ID for "Fellowship/Gift" plan

## 4. SendGrid (Email)
Required for sending login links and gift notifications.
- `SENDGRID_API_KEY`: `SG....`
- `FROM_EMAIL`: `hello@curiouskelly.com`

**Template IDs (Create Dynamic Templates in SendGrid):**
*Note: You can use a placeholder ID if you haven't created the template yet, but emails won't send correctly.*
- `TEMPLATE_WAITLIST`
- `TEMPLATE_GIFT_RECIPIENT`
- `TEMPLATE_GIFTER_CONFIRM`
- `TEMPLATE_WELCOME`

## How to Deploy to Railway
1. Connect your GitHub repo.
2. Point the "Root Directory" to `curious-kellly/backend`.
3. Add the PostgreSQL plugin.
4. Paste these variables into the "Variables" tab.


























