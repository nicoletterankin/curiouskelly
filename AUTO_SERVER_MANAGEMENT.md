# 🤖 Automatic Server Management

**You never have to think about servers again.** Everything is automated.

## What Happens Automatically

### When You Open VS Code
- ✅ Infrastructure (Postgres, Redis, Meilisearch, ClickHouse) starts automatically
- ✅ Dev servers (Gateway API + Classroom WebSocket) start automatically  
- ✅ HTML files auto-open in your browser when created/modified
- ✅ API docs open automatically at `http://localhost:4000/docs`

### When You Create/Modify HTML Files
- ✅ Files automatically open in your default browser
- ✅ Works with Live Server extension (right-click → "Open with Live Server")
- ✅ Works with file:// protocol (direct file opening)

## How It Works

### VS Code Integration
1. **Auto-Start Task**: Runs `🚀 Start Everything (Auto-Managed)` when you open the workspace
2. **File Watcher**: Background script watches for HTML file changes
3. **Launch Configs**: Debug configurations ready for F5 debugging

### Manual Control (If Needed)
- **Start Everything**: `Ctrl+Shift+P` → "Tasks: Run Task" → "🚀 Start Everything (Auto-Managed)"
- **Stop Everything**: `Ctrl+Shift+P` → "Tasks: Run Task" → "Stop All Servers"
- **Start Individual Services**: Use task menu to start Gateway or Classroom separately

## Server Ports

| Service | Port | URL |
|---------|------|-----|
| Gateway API | 4000 | http://localhost:4000/docs |
| Classroom WebSocket | 4100 | ws://localhost:4100 |
| Postgres | 5432 | localhost:5432 |
| Redis | 6379 | localhost:6379 |
| Meilisearch | 7700 | http://localhost:7700 |
| ClickHouse | 8123 | http://localhost:8123 |

## HTML File Auto-Opening

The system watches for:
- New HTML files created anywhere in the repo
- HTML files that are modified/saved
- Debounced to avoid opening multiple times for rapid saves

**Works with:**
- VS Code Live Server extension (recommended)
- Direct file:// protocol
- Any HTML file in the workspace

## Troubleshooting

### Servers Don't Start Automatically
1. Check Docker Desktop is running
2. Verify `pnpm` is installed: `pnpm --version`
3. Run manually: `.\scripts\dev-server.ps1 -Target stack`

### HTML Files Don't Auto-Open
1. Check VS Code has permission to open URLs
2. Try right-clicking HTML file → "Open with Live Server"
3. Manually open: Right-click → "Reveal in File Explorer" → double-click

### Port Already in Use
- Stop existing servers: `.\scripts\dev-server.ps1 -Action stop`
- Or change ports in `docker-compose.dev.yml` and `apps/*/src/index.ts`

## Disabling Auto-Start

If you want to disable automatic server startup:

1. Open `.vscode/tasks.json`
2. Find the `🚀 Start Everything (Auto-Managed)` task
3. Remove or comment out the `"runOptions": { "runOn": "folderOpen" }` line

## Unity Workflow

When working in Unity:
- Servers stay running in the background
- HTML files still auto-open when you create them
- No need to stop/start anything - it's all managed automatically

---

**Remember**: You never have to think about servers. Just open VS Code and start coding. Everything else is handled automatically. 🚀








