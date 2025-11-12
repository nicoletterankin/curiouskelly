## Daily Lesson Platform Backend

This document outlines the new TypeScript backend stack that powers the daily-lesson experience with hourly live classes.

### Monorepo Layout

```
apps/
  gateway/         # REST API (Fastify)
  classroom/       # WebSocket & LiveKit control plane
packages/
  config/          # Zod environment loader
  database/        # Prisma client wrapper
  logger/          # Pino logger with OTEL correlation
  testing/         # Vitest + Testcontainers helpers
  types/           # Shared Zod schemas & DTOs
services/
  auth/            # Passkeys, magic links, JWT, device binding
  entitlements/    # Plans, feature caching, Stripe sync
  lessons/         # Pack signer and manifest retrieval
  schedule/        # Hourly slot generation, waitlists
  classroom/       # LiveKit token minting & moderation queues
  payments/        # Stripe webhook handler & dunning jobs
  notifications/   # Email/push dispatch with quiet hours
  search/          # Meilisearch indexing
  telemetry/       # RUM + server metrics sink (ClickHouse)
  ops/             # Staff SSO utilities & internal tooling
scripts/
  schedule-seed.ts # Fills two-week rolling schedule
  lessons-build.ts # Compiles markdown content packs
tests/
  (reserved for unit/integration/e2e suites)
```

### Data Persistence

- **Postgres (Prisma)** – canonical store for users, schedule, sessions, replays, feedback, notifications, audit events.
- **Redis (Upstash/Valkey)** – entitlements cache, rate limiting, schedule reservation counters, moderation queues.
- **BullMQ queues** – `entitlements.sync`, `schedule.seed`, `payments.dunning`, `notifications.dispatch`, `classroom.moderation`.
- **R2 (S3-compatible)** – immutable lesson packs and replay recordings.
- **Meilisearch** – topic and lesson discovery.
- **ClickHouse** – RUM + server telemetry with privacy filters.

### API Surface

- `GET /v1/lessons/today` → `LessonsService.getTodayManifest`
- `GET /v1/schedule/next` → `ScheduleService.getNextSlots`
- `POST /v1/join` → entitlement check, seat reservation, LiveKit token minting or HLS overflow
- `POST /v1/sessions/:id/attendance` → idempotent attendance tracking
- `GET /v1/sessions/:id/replay` → signed replay URL
- `GET /v1/search` → Meilisearch query
- `POST /v1/feedback` → CSAT / NPS capture
- Webhooks: `/webhooks/stripe`, `/webhooks/livekit`
- WebSocket: `/v1/classroom/:sessionId` (events: presence, chat, polls, moderation)

### Observability

- Pino logger enriched with OpenTelemetry trace/span IDs.
- OTLP-compatible instrumentation hook (reserved in `@acme/logger`).
- Metrics focus: join latency, room utilization, overflow activations, queue depth, abandonment.
- Privacy-first analytics: RUM disabled by default (`RUM_ENABLED=false`), ingestion gated by explicit consent flag.

### Local Development

1. **Install dependencies**
   ```bash
   pnpm install
   ```
2. **Database**
   ```bash
   pnpm dlx prisma migrate dev
   ```
3. **Start services**
   ```bash
   pnpm --filter @acme/gateway dev
   pnpm --filter @acme/classroom dev
   ```
4. **Seed schedule**
   ```bash
   pnpm --filter @acme/scripts schedule:seed
   ```

### Next Steps

- Flesh out passkey + magic link flows in `@acme/service-auth`.
- Implement LiveKit webhook verification and overflow HLS playback.
- Add Vitest unit tests and integration suites using `@acme/testing`.
- Wire CI workflows (`.github/workflows/`) per deployment target.
- Define Terraform/Pulumi modules under `infra/` (Neon, Upstash, R2, Meilisearch, ClickHouse).

