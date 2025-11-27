# Deployment Guide (Stupid Easy Version)

## Part 1: Backend (Railway)
Since you are already on the "New Project" screen:

1.  **Select "GitHub Repository"**.
2.  Choose this repository (`curiouskelly`).
3.  **Important:** Click "Settings" (or "Config") for the service before deploying, or go to "Settings" after it fails.
    *   Find **Root Directory**.
    *   Set it to: `curious-kellly/backend`
    *   Click Save/Deploy.
4.  **Add Database:**
    *   Right-click the canvas (or click "New") -> Database -> **PostgreSQL**.
    *   It will auto-link `DATABASE_URL`.
5.  **Add Secrets:**
    *   Go to the "Variables" tab of your Backend service.
    *   Copy/Paste the values from `SECRETS_NEEDED.md` (Stripe keys, SendGrid, etc.).
6.  **Get the URL:**
    *   Go to "Settings" -> "Networking" -> Generate Domain.
    *   Copy this URL (e.g., `https://curious-kelly-production.up.railway.app`).

## Part 2: Frontend (Vercel)
You already have Vercel connected. We just need to point it to the new, clean OS.

1.  Go to your Vercel Project Settings -> **General**.
2.  Find **Root Directory**.
3.  Change it to: `curious-kellly/lesson-player-v2`
4.  Click Save.
5.  Go to **Deployments** -> Redeploy the latest commit.

## Part 3: Connect Them
1.  Once you have the Railway URL (from Part 1), go to your code:
    *   File: `curious-kellly/lesson-player-v2/js/app.js`
    *   Line ~8: Update `API_URL` with your real Railway URL.
2.  Commit and Push. Vercel will update automatically.

**That's it. You're live.**







