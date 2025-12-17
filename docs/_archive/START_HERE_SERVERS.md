# 🚀 Server Management - You're All Set!

## TL;DR

**Everything is automated. Just open VS Code and start coding.**

- ✅ Servers start automatically when you open the workspace
- ✅ HTML files auto-open when you create/modify them  
- ✅ Works seamlessly with Live Server (right-click → "Open with Live Server")
- ✅ All infrastructure managed automatically

## What Happens Automatically

1. **When VS Code Opens**: Infrastructure + dev servers start automatically
2. **When You Create HTML**: File automatically opens in your browser
3. **When You Modify HTML**: File automatically refreshes/opens
4. **When You Close VS Code**: Servers stay running (you can stop manually if needed)

## Manual Control (Rarely Needed)

### Start Everything
```powershell
.\scripts\dev-server.ps1 -Target stack
```

### Stop Everything  
```powershell
.\scripts\dev-server.ps1 -Action stop
```

### VS Code Tasks
- `Ctrl+Shift+P` → "Tasks: Run Task" → Choose:
  - `🚀 Start Everything (Auto-Managed)` - Starts all servers
  - `Stop All Servers` - Stops everything
  - `Start Gateway API` - Just the API
  - `Start Classroom WebSocket` - Just WebSocket

## HTML Files

**Two ways to open HTML:**

1. **Automatic** (recommended): Just save/create the file - it opens automatically
2. **Live Server**: Right-click HTML → "Open with Live Server" (works great!)

## Unity Workflow

When working in Unity:
- Servers run in background - no need to stop them
- HTML files still auto-open when you create them
- Everything stays out of your way

## Troubleshooting

**Servers don't start?**
- Check Docker Desktop is running
- Run manually: `.\scripts\dev-server.ps1 -Target stack`

**HTML doesn't auto-open?**
- Try right-click → "Open with Live Server"
- Check browser isn't blocking popups

**Port in use?**
- Stop existing: `.\scripts\dev-server.ps1 -Action stop`
- Or change ports in config files

---

**That's it!** You never have to think about servers again. Just code. 🎉








