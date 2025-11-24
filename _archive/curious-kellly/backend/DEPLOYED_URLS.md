# Deployed Backend URLs

## Production (Render)
- **Base URL**: https://YOUR_RENDER_URL.onrender.com
- **Health**: https://YOUR_RENDER_URL.onrender.com/health
- **OpenAI Test**: https://YOUR_RENDER_URL.onrender.com/api/realtime/test
- **Kelly Endpoint**: https://YOUR_RENDER_URL.onrender.com/api/realtime/kelly
- **Safety Moderate**: https://YOUR_RENDER_URL.onrender.com/api/safety/moderate
- **Dashboard**: https://dashboard.render.com

## GitHub
- **Repository**: https://github.com/YOUR_USERNAME/curious-kellly
- **Backend Code**: https://github.com/YOUR_USERNAME/curious-kellly/tree/main/curious-kellly/backend

## Status
- Deployed: October 29, 2025
- Version: 0.2.0 (Day 3 - Lesson System)
- Status: 🟢 Live
- Last Updated: Day 3 deployment

## New Endpoints (Day 3)
- **/api/lessons/today** - Get today's universal lesson
- **/api/lessons/today/:age** - Get today's lesson for specific age
- **/api/lessons/:id** - Get lesson by ID
- **/api/lessons/:id/age/:age** - Get lesson by ID for specific age
- **/api/sessions/start** - Start a new lesson session
- **/api/sessions/:id/progress** - Update session progress
- **/api/sessions/:id/complete** - Complete session
- **/api/sessions/:id/stats** - Get session statistics

## Quick Commands

### Test Health
```bash
curl https://YOUR_RENDER_URL.onrender.com/health
```

### View Logs
Open: https://dashboard.render.com → curious-kellly-backend → Logs

### Redeploy
```bash
git add .
git commit -m "Update"
git push
# Auto-deploys!
```

