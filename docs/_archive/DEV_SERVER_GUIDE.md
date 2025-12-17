## Full-Stack Dev Server Quick Start

This guide explains how to bring up the Daily Lesson Platform stack locally with a single command and shut it down just as easily. It follows the priorities in `CLAUDE.md` (fast startup, predictable experience, no surprises).

### Prerequisites

1. **Docker Desktop** (latest stable) – provides Postgres, Redis, Meilisearch, and ClickHouse containers.
2. **Node.js 20.11+** (matches `engines.node` in `package.json`).
3. **pnpm 9.9+** – install via `corepack enable` or see [pnpm.io/installation](https://pnpm.io/installation).
4. Optional: Windows Terminal / PowerShell 7 for better Ctrl+C handling.

Run `pnpm install` once from the repo root to hydrate all workspaces.

### Start everything (one command)

```powershell
cd C:\Users\user\UI-TARS-desktop
.\scripts\dev-server.ps1
```

What happens:

- Docker containers start for Postgres (`5432`), Redis (`6379`), Meilisearch (`7700`), and ClickHouse (`8123`).
- Health checks ensure each port is reachable before the app starts.
- `pnpm dev:stack` launches both runtime servers:
  - API Gateway (`http://localhost:4000`, OpenAPI docs at `/docs`).
  - Classroom WebSocket control plane (`ws://localhost:4100/v1/classroom/:sessionId`).
- Press **Ctrl+C once** to stop the Node servers; the script automatically shuts down Docker unless `-KeepDependencies` is supplied.

### Stopping services

- While the servers are running in the same terminal, press **Ctrl+C**.
- To explicitly tear down Docker later (e.g., after using `-KeepDependencies`), run:

```powershell
.\scripts\dev-server.ps1 -Action stop
```

### Script switches

| Example | What it does |
| --- | --- |
| `.\scripts\dev-server.ps1 -Target gateway` | Starts only the Fastify HTTP gateway (port 4000) plus infra. |
| `.\scripts\dev-server.ps1 -Target classroom` | Starts only the WebSocket classroom service (port 4100) plus infra. |
| `.\scripts\dev-server.ps1 -Target deps` | Starts Docker infrastructure only; press Ctrl+C to stop (or add `-KeepDependencies`). |
| `.\scripts\dev-server.ps1 -KeepDependencies` | Leaves Docker containers running after you stop the Node servers. Tear down with `-Action stop` later. |
| `.\scripts\dev-server.ps1 -SkipDependencies` | Assume Postgres/Redis/Meilisearch/ClickHouse are already running and only start the selected Node server(s). |
| `.\scripts\dev-server.ps1 -Action status` | Show `docker compose ps` for the dev stack. |

### Verification checklist

1. `http://localhost:4000/health` → `{"status":"ok"}`.
2. `http://localhost:4000/docs` renders the Swagger UI for the gateway routes.
3. `http://localhost:4000/v1/lessons/today?topic=gratitude&locale=en-US` returns a JSON payload (defaults require seeded data).
4. `http://localhost:4000/webhooks/livekit` responds with 200 when posting a dummy payload (confirms env + JWT secret wiring).
5. Optional WebSocket smoke test: `npx wscat -c ws://localhost:4100/v1/classroom/demo-session`.

### Troubleshooting

- **Docker missing**: Script stops with `Missing dependency: docker`. Install Docker Desktop and rerun.
- **Ports already used**: Free conflicting services (5432/6379/7700/8123) or run `-SkipDependencies` if you intentionally run custom instances.
- **pnpm not found**: Run `corepack enable pnpm` or install manually, then retry.
- **Need clean volumes**: `docker compose -f docker-compose.dev.yml down -v` removes persisted data; next run reseeds empty databases.

### Files involved

- `docker-compose.dev.yml` – declares local infra containers.
- `scripts/dev-server.ps1` – orchestration + health checks.
- `package.json` scripts:
  - `dev:gateway`, `dev:classroom` → single services.
  - `dev:stack` → parallel gateway + classroom.

This workflow keeps setup aligned with the `BUILD_PLAN.md` requirement for fast local validation while satisfying `CLAUDE.md`'s zero-surprise operations contract.








