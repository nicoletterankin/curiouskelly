# Deployment Guide - Curious Kellly Backend

## 🚀 Deploy to Render.com

### Prerequisites
- [ ] GitHub account
- [ ] Render.com account (free) - https://render.com
- [ ] OpenAI API key

---

## Step-by-Step Deployment

### 1. Push Code to GitHub (If Not Already)

```bash
cd C:\Users\user\UI-TARS-desktop
git init
git add curious-kellly/backend/
git commit -m "Initial backend deployment"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/curious-kellly.git
git push -u origin main
```

---

### 2. Sign Up for Render

1. Go to https://render.com
2. Click "Get Started for Free"
3. Sign up with GitHub (easiest)
4. Authorize Render to access your repositories

---

### 3. Create New Web Service

1. Click "New +"
2. Select "Web Service"
3. Connect your GitHub repository
4. Select the `curious-kellly` repo

**Configuration:**
- **Name**: `curious-kellly-backend`
- **Root Directory**: `curious-kellly/backend`
- **Environment**: `Node`
- **Build Command**: `npm install`
- **Start Command**: `npm start`
- **Plan**: `Free`

Click "Create Web Service"

---

### 4. Add Environment Variables

In the Render dashboard:

1. Go to "Environment" tab
2. Add these variables:

```
NODE_ENV = production
PORT = 10000
OPENAI_API_KEY = sk-proj-YOUR_KEY_HERE
OPENAI_REALTIME_MODEL = gpt-4o-realtime-preview-2024-10-01
LOG_LEVEL = info
```

3. Click "Save Changes"

The service will automatically redeploy.

---

### 5. Wait for Deployment

- First deploy takes 2-5 minutes
- Watch the logs in real-time
- You'll see: "Build successful" → "Deploy live"

---

### 6. Get Your Live URL

Your backend will be live at:
```
https://curious-kellly-backend.onrender.com
```

(Or similar - Render will show you the exact URL)

---

## Testing Your Deployment

### Health Check
```bash
curl https://curious-kellly-backend.onrender.com/health
```

Expected:
```json
{
  "status": "ok",
  "service": "curious-kellly-backend",
  "environment": "production"
}
```

### Test OpenAI
```bash
curl https://curious-kellly-backend.onrender.com/api/realtime/test
```

### Test Kelly
```bash
curl -X POST https://curious-kellly-backend.onrender.com/api/realtime/kelly \
  -H "Content-Type: application/json" \
  -d '{"age":35,"topic":"leaves","message":"Why do leaves change color?"}'
```

---

## Alternative: Deploy to Railway

If you prefer Railway.app:

1. Go to https://railway.app
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select `curious-kellly/backend`
5. Add environment variables (same as above)
6. Deploy!

Railway gives you: `https://curious-kellly-backend.up.railway.app`

---

## Monitoring & Logs

### View Logs
- **Render**: Dashboard → Logs tab
- **Railway**: Dashboard → Deployments → View Logs

### Metrics
- **Render**: Dashboard → Metrics tab
- **Railway**: Dashboard → Metrics

### Alerts
Set up email alerts for:
- Service down
- High error rate
- High latency

---

## Updating Deployment

Every time you push to GitHub:
```bash
git add .
git commit -m "Update backend"
git push
```

Render/Railway will automatically redeploy!

---

## Custom Domain (Optional)

### Add Your Domain
1. In Render: Settings → Custom Domain
2. Add: `api.curiousskellly.com`
3. Update DNS with provided CNAME
4. SSL certificate auto-provisioned

---

## Troubleshooting

### "Build failed"
- Check Node version (should be 18+)
- Verify `package.json` is valid
- Check build logs for errors

### "Service unhealthy"
- Check environment variables set correctly
- Verify OPENAI_API_KEY is valid
- Check logs for startup errors

### "502 Bad Gateway"
- Service starting up (wait 30 seconds)
- Check if PORT is set correctly
- Restart service

---

## Production Checklist

Before going live:
- [ ] Environment variables set
- [ ] Health endpoint returns 200
- [ ] OpenAI test endpoint works
- [ ] Safety tests pass
- [ ] Logs show no errors
- [ ] SSL certificate active
- [ ] Custom domain configured (optional)
- [ ] Monitoring alerts set up

---

**Status**: Ready to deploy  
**Estimated Time**: 30 minutes  
**Cost**: $0 (Free tier)















