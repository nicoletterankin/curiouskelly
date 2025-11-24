# Curious Kelly Backend API

Backend service for Curious Kelly Christmas launch.

## Setup

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Environment

Copy `env.template` to `.env` and fill in your values:

```bash
cp env.template .env
```

Required environment variables:
- `DATABASE_URL` - PostgreSQL connection string
- `STRIPE_SECRET_KEY` - Stripe API secret key
- `STRIPE_WEBHOOK_SECRET` - Stripe webhook signing secret
- `SENDGRID_API_KEY` - SendGrid API key
- Template IDs for all 14 email templates

See `env.template` for complete list.

### 3. Set Up Database

Create PostgreSQL database:

```bash
createdb curious_kelly
```

Run migrations:

```bash
npm run migrate
```

### 4. Start Development Server

```bash
npm run dev
```

Server will start on http://localhost:3000

## API Endpoints

### Health Check
- `GET /health` - Check server status

### Checkout
- `POST /api/checkout/create-session` - Create Stripe checkout session
  - Body: `{ plan, customerEmail, recipientEmail?, giftMessage?, gifterName? }`

### Gifts
- `POST /api/gifts/create` - Create gift record (internal, called from webhook)
- `GET /api/gifts/verify/:code` - Verify gift code validity
- `POST /api/gifts/redeem` - Redeem gift code
  - Body: `{ giftCode, userEmail, userName? }`

### Users
- `POST /api/users/create` - Create new user
  - Body: `{ email, name?, age?, plan? }`
- `GET /api/users/:id` - Get user by ID
- `PUT /api/users/:id` - Update user profile
  - Body: `{ name?, age? }`

### Lessons
- `GET /api/lessons/calendar` - Get full 365-day calendar
- `GET /api/lessons/day/:day` - Get specific lesson (1-365)
- `POST /api/lessons/complete` - Mark lesson as completed
  - Body: `{ userId, lessonDay, lessonId, durationSeconds?, ageVariant? }`
- `GET /api/lessons/user/:userId/progress` - Get user's progress

### Webhooks
- `POST /webhook` - Stripe webhook handler (requires raw body)

## Development

### Run with auto-reload

```bash
npm run dev
```

### Run tests

```bash
npm test
```

## Deployment

### Environment Variables

Set all environment variables in your hosting platform:
- Vercel: Project Settings → Environment Variables
- Railway: Project → Variables
- Heroku: Settings → Config Vars

### Deploy to Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Add PostgreSQL
railway add postgresql

# Deploy
railway up
```

### Deploy to Heroku

```bash
# Install Heroku CLI
# Login
heroku login

# Create app
heroku create curious-kelly-api

# Add PostgreSQL
heroku addons:create heroku-postgresql:mini

# Set environment variables
heroku config:set STRIPE_SECRET_KEY=sk_live_xxxxx
# ... (set all other env vars)

# Deploy
git push heroku main

# Run migrations
heroku run npm run migrate
```

## Stripe Webhook Setup

1. Go to Stripe Dashboard → Developers → Webhooks
2. Add endpoint: `https://your-api.com/webhook`
3. Select events:
   - `checkout.session.completed`
   - `customer.subscription.created`
   - `customer.subscription.deleted`
   - `invoice.payment_succeeded`
   - `invoice.payment_failed`
4. Copy webhook signing secret to `.env`:
   ```
   STRIPE_WEBHOOK_SECRET=whsec_xxxxx
   ```

## Testing

### Test Stripe Checkout

```bash
# Create test checkout session
curl -X POST http://localhost:3000/api/checkout/create-session \
  -H "Content-Type: application/json" \
  -d '{
    "plan": "gift",
    "customerEmail": "test@example.com",
    "recipientEmail": "recipient@example.com",
    "gifterName": "John Doe",
    "giftMessage": "Merry Christmas!"
  }'
```

### Test Gift Redemption

```bash
# Verify gift code
curl http://localhost:3000/api/gifts/verify/CK-TEST1-TEST1

# Redeem gift
curl -X POST http://localhost:3000/api/gifts/redeem \
  -H "Content-Type: application/json" \
  -d '{
    "giftCode": "CK-TEST1-TEST1",
    "userEmail": "user@example.com",
    "userName": "Jane Smith"
  }'
```

### Test Webhook Locally

Use Stripe CLI to forward webhooks to localhost:

```bash
# Install Stripe CLI
# Login
stripe login

# Forward webhooks
stripe listen --forward-to localhost:3000/webhook

# Trigger test event
stripe trigger checkout.session.completed
```

## Monitoring

### Logs

```bash
# Heroku logs
heroku logs --tail

# Railway logs
railway logs
```

### Database

```bash
# Connect to database
psql $DATABASE_URL

# Check tables
\dt

# Check gift codes
SELECT * FROM gifts ORDER BY created_at DESC LIMIT 10;

# Check users
SELECT * FROM users ORDER BY created_at DESC LIMIT 10;
```

## Support

For issues or questions:
- Email: hello@curiouskelly.com
- Documentation: See `IMPLEMENTATION_GUIDE.md`
