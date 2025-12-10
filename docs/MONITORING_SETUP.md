# Monitoring & Observability Setup Guide
**Curious Kelly Platform - Production Monitoring**

---

## 🎯 Overview

This guide covers setting up comprehensive monitoring, error tracking, and observability for the Curious Kelly platform. Proper monitoring is critical for production operations.

---

## 📊 Monitoring Stack

### Required Components:
1. **Error Tracking** - Sentry (recommended)
2. **Uptime Monitoring** - UptimeRobot or Pingdom
3. **Application Performance Monitoring (APM)** - Vercel Analytics or Cloudflare Analytics
4. **Log Aggregation** - Cloudflare Logs or Datadog

---

## 1. 🔴 Error Tracking (Sentry)

### Why Sentry?
- Real-time error tracking
- Source map support
- Performance monitoring
- Release tracking
- User context

### Setup Steps:

#### Step 1: Create Sentry Account
1. Go to https://sentry.io/signup/
2. Create account or sign in
3. Create new project: "Curious Kelly"
4. Select platform: **JavaScript** (Astro/TypeScript)

#### Step 2: Install Sentry SDK

```bash
cd daily-lesson-marketing
pnpm add @sentry/astro @sentry/browser
```

#### Step 3: Configure Sentry

Create `src/lib/sentry.ts`:

```typescript
import * as Sentry from "@sentry/astro";

export function initSentry() {
  if (import.meta.env.PUBLIC_SENTRY_DSN) {
    Sentry.init({
      dsn: import.meta.env.PUBLIC_SENTRY_DSN,
      environment: import.meta.env.MODE,
      tracesSampleRate: 1.0, // 100% in dev, lower in prod
      beforeSend(event, hint) {
        // Filter out sensitive data
        if (event.request) {
          delete event.request.cookies;
          delete event.request.headers?.authorization;
        }
        return event;
      },
    });
  }
}
```

#### Step 4: Add to Astro Config

Update `astro.config.mjs`:

```javascript
import { defineConfig } from 'astro/config';
import sentry from '@sentry/astro';

export default defineConfig({
  integrations: [
    sentry({
      dsn: import.meta.env.PUBLIC_SENTRY_DSN,
      sourceMapsUploadOptions: {
        org: 'your-org',
        project: 'curious-kelly',
        authToken: import.meta.env.SENTRY_AUTH_TOKEN,
      },
    }),
  ],
});
```

#### Step 5: Add Environment Variables

Add to `.env`:
```bash
PUBLIC_SENTRY_DSN=https://your-key@sentry.io/your-project-id
SENTRY_AUTH_TOKEN=your-auth-token
```

Add to GitHub Secrets:
- `SENTRY_AUTH_TOKEN` (for source maps upload)

#### Step 6: Initialize in Layout

Update `src/layouts/SiteLayout.astro`:

```astro
---
import { initSentry } from '@lib/sentry';

if (import.meta.env.PROD) {
  initSentry();
}
---

<!-- rest of layout -->
```

### Testing Sentry:

```typescript
// Test error tracking
Sentry.captureException(new Error("Test error"));
```

---

## 2. ⏱️ Uptime Monitoring (UptimeRobot)

### Why UptimeRobot?
- Free tier: 50 monitors
- 5-minute check intervals
- Email/SMS alerts
- Status page support

### Setup Steps:

#### Step 1: Create Account
1. Go to https://uptimerobot.com/
2. Sign up for free account
3. Verify email

#### Step 2: Add Monitors

**Critical Endpoints to Monitor:**

1. **Homepage**
   - URL: `https://curiouskelly.com`
   - Type: HTTP(s)
   - Interval: 5 minutes
   - Alert Contacts: Your email

2. **API Health Check**
   - URL: `https://curiouskelly.com/api/health`
   - Type: HTTP(s)
   - Interval: 5 minutes
   - Expected Status: 200

3. **Lead Form Endpoint**
   - URL: `https://curiouskelly.com/api/lead`
   - Type: HTTP(s) - Keyword
   - Keyword: `"success":true`
   - Interval: 15 minutes

4. **Database Backup Status**
   - Check GitHub Actions workflow status
   - Use GitHub API: `https://api.github.com/repos/OWNER/REPO/actions/workflows/database-backup.yml/runs`
   - Type: HTTP(s) - Keyword
   - Keyword: `"conclusion":"success"`

#### Step 3: Configure Alerts
- Email alerts: ✅ Enabled
- SMS alerts: Configure if needed
- Alert when: Down for 1 check

#### Step 4: Create Status Page (Optional)
1. Go to "My Settings" → "Public Status Pages"
2. Create status page
3. Add monitors to page
4. Share URL: `status.curiouskelly.com`

---

## 3. 📈 Application Performance Monitoring (APM)

### Option A: Vercel Analytics (If Using Vercel)

#### Setup:
1. Go to Vercel Dashboard → Project Settings → Analytics
2. Enable Web Analytics
3. Add to `astro.config.mjs`:

```javascript
import vercel from '@astrojs/vercel/analytics';

export default defineConfig({
  integrations: [
    vercel({
      mode: 'production', // Only track in production
    }),
  ],
});
```

### Option B: Cloudflare Analytics (If Using Cloudflare)

#### Setup:
1. Go to Cloudflare Dashboard → Analytics & Logs → Web Analytics
2. Enable Web Analytics
3. Add script to layout (auto-injected by Cloudflare)

### Option C: Custom RUM (Already Implemented)

Your existing `/api/rum` endpoint can be enhanced:

```typescript
// src/lib/rum.ts
export function initRUM() {
  if (import.meta.env.PUBLIC_RUM_ENABLED !== 'true') return;

  // Track Core Web Vitals
  new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      if (entry.entryType === 'largest-contentful-paint') {
        sendMetric('LCP', entry.renderTime || entry.loadTime);
      }
      if (entry.entryType === 'first-input') {
        sendMetric('FID', entry.processingStart - entry.startTime);
      }
    }
  }).observe({ entryTypes: ['largest-contentful-paint', 'first-input'] });

  // Track CLS
  let clsValue = 0;
  new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      if (!entry.hadRecentInput) {
        clsValue += entry.value;
      }
    }
    sendMetric('CLS', clsValue);
  }).observe({ entryTypes: ['layout-shift'] });
}

function sendMetric(name: string, value: number) {
  fetch('/api/rum', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ metric: name, value, timestamp: Date.now() }),
  }).catch(() => {}); // Fail silently
}
```

---

## 4. 📝 Log Aggregation

### Option A: Cloudflare Logs (If Using Cloudflare)

#### Setup:
1. Go to Cloudflare Dashboard → Analytics & Logs → Logs
2. Enable Logpush
3. Configure destination (S3, R2, Datadog, etc.)

### Option B: Vercel Logs (If Using Vercel)

#### Access:
1. Go to Vercel Dashboard → Project → Functions
2. Click on function (e.g., `/api/lead`)
3. View real-time logs

### Option C: Custom Log Aggregation

Create `src/lib/logger.ts`:

```typescript
export function log(level: 'info' | 'warn' | 'error', message: string, data?: any) {
  const logEntry = {
    timestamp: new Date().toISOString(),
    level,
    message,
    data,
    url: typeof window !== 'undefined' ? window.location.href : '',
    userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : '',
  };

  // Send to logging service
  if (import.meta.env.PUBLIC_LOG_ENDPOINT) {
    fetch(import.meta.env.PUBLIC_LOG_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(logEntry),
    }).catch(() => {});
  }

  // Also log to console in development
  if (import.meta.env.DEV) {
    console[level](logEntry);
  }
}
```

---

## 5. 🚨 Alerting Configuration

### Alert Channels:

1. **Email Alerts**
   - Sentry: Automatic for errors
   - UptimeRobot: Configured in monitors
   - GitHub Actions: Workflow failure notifications

2. **Slack Integration (Optional)**
   - Sentry: Add Slack webhook
   - UptimeRobot: Add Slack integration
   - GitHub Actions: Add Slack notification step

3. **PagerDuty (For Critical Alerts)**
   - Set up PagerDuty account
   - Integrate with Sentry
   - Configure on-call rotation

### Alert Rules:

**Critical (Immediate Response):**
- Production site down
- Database backup failure
- Payment processing errors
- Security incidents

**High (Response within 1 hour):**
- API errors > 5%
- Performance degradation
- High error rate

**Medium (Response within 4 hours):**
- Non-critical feature failures
- Performance regressions
- Warning-level issues

---

## 6. 📊 Dashboards

### Recommended Dashboards:

1. **Operations Dashboard**
   - Uptime status
   - Error rate
   - Response times
   - Active users

2. **Performance Dashboard**
   - Core Web Vitals
   - API response times
   - Database query times
   - Cache hit rates

3. **Business Metrics Dashboard**
   - Lead submissions
   - Conversion rates
   - User engagement
   - Revenue metrics

### Tools:
- **Grafana** (self-hosted or cloud)
- **Datadog** (paid)
- **Sentry Performance** (included)
- **Vercel Analytics** (if using Vercel)

---

## 7. ✅ Setup Checklist

### Week 1 (Critical):
- [ ] Set up Sentry account and integrate
- [ ] Add Sentry DSN to environment variables
- [ ] Create UptimeRobot account
- [ ] Add monitors for critical endpoints
- [ ] Configure email alerts

### Week 2 (High Priority):
- [ ] Enable APM (Vercel/Cloudflare Analytics)
- [ ] Set up log aggregation
- [ ] Create operations dashboard
- [ ] Test alerting (trigger test alerts)

### Week 3 (Medium Priority):
- [ ] Set up Slack integration
- [ ] Create performance dashboard
- [ ] Configure PagerDuty (if needed)
- [ ] Document on-call procedures

---

## 8. 🔧 Maintenance

### Daily:
- Review error reports in Sentry
- Check uptime status
- Review performance metrics

### Weekly:
- Review error trends
- Check alert effectiveness
- Update dashboards if needed

### Monthly:
- Review monitoring costs
- Optimize alert thresholds
- Update documentation

---

## 📚 Resources

- [Sentry Documentation](https://docs.sentry.io/)
- [UptimeRobot Documentation](https://uptimerobot.com/api/)
- [Vercel Analytics](https://vercel.com/docs/analytics)
- [Cloudflare Analytics](https://developers.cloudflare.com/analytics/)

---

## 🆘 Support

For questions or issues:
1. Check this documentation
2. Review `RUNBOOK.md` for incident procedures
3. Check `SYSTEMS_CHECK_REPORT.md` for system status

---

**Last Updated:** December 2025  
**Next Review:** After monitoring is fully implemented
















