# Curious Kelly Lessons API Worker

Cloudflare Worker that serves lesson data from D1 as a Supabase mirror.
If Supabase goes down, D1 serves lessons from the edge.

## Setup

### 1. Create D1 Database

```bash
cd infrastructure/cloudflare/lessons-api-worker

# Install dependencies
npm install

# Create the D1 database
npx wrangler d1 create kelly-lessons-mirror
```

Copy the `database_id` from the output and update `wrangler.toml`:

```toml
[[d1_databases]]
binding = "DB"
database_name = "kelly-lessons-mirror"
database_id = "YOUR_ACTUAL_DATABASE_ID"  # <-- Paste here
```

### 2. Apply Schema

```bash
# Apply schema to local D1 (for testing)
npx wrangler d1 execute kelly-lessons-mirror --local --file=./schema.sql

# Apply schema to production D1
npx wrangler d1 execute kelly-lessons-mirror --file=./schema.sql
```

### 3. Deploy Worker

```bash
# Deploy to production
npx wrangler deploy

# Or deploy to staging
npx wrangler deploy --env staging
```

### 4. Sync Data from Supabase

Run the sync script to populate D1 with Supabase data:

```bash
# From repo root
node scripts/sync-to-d1.js
```

Set up a cron job or GitHub Action to run this daily.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Database health check |
| `GET /lesson/:day` | Get lesson with atoms/shards |
| `GET /lessons` | List all lessons |
| `GET /sync/status` | Last sync status |

### Example Requests

```bash
# Health check
curl https://curiouskelly-lessons.YOUR_SUBDOMAIN.workers.dev/health

# Get lesson 1
curl "https://curiouskelly-lessons.YOUR_SUBDOMAIN.workers.dev/lesson/1?archetype=The%20Scientist&region=adult"

# List all lessons
curl https://curiouskelly-lessons.YOUR_SUBDOMAIN.workers.dev/lessons
```

## Environment Variables

The sync script requires these environment variables:

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Supabase service role key |
| `CF_ACCOUNT_ID` | Cloudflare account ID |
| `CF_API_TOKEN` | Cloudflare API token with D1 permissions |
| `D1_DATABASE_ID` | The database_id from step 1 |

## Local Development

```bash
# Start local dev server
npx wrangler dev

# Test with local D1
curl http://localhost:8787/health
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│     Browser     │     │   KellyLoader   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         │  1. Try Supabase      │
         │◄──────────────────────│
         │                       │
         │  2. If fail, try D1   │
         │◄──────────────────────│
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│    Supabase     │     │ Cloudflare D1   │
│   (Primary)     │     │   (Mirror)      │
└─────────────────┘     └─────────────────┘
```

## Sync Schedule

The sync script should run daily to keep D1 in sync with Supabase:
- Run after any content updates
- Recommended: 3 AM UTC daily
- Average sync time: ~30 seconds for 365 lessons

## Monitoring

Check sync status at `/sync/status`:

```json
{
  "synced": true,
  "lastSyncAt": "2025-12-12T03:00:00.000Z",
  "counts": {
    "lessons": 365,
    "atoms": 21915,
    "shards": 38700
  },
  "syncSource": "supabase",
  "syncDurationMs": 28500
}
```

