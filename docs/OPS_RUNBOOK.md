# Curious Kelly Operations Runbook

Quick reference for common issues after launch.

## Quick Links

- **Health Check**: https://curiouskelly.com/api/health
- **Stats**: `curl -H "Authorization: Bearer $CRON_SECRET" https://curiouskelly.com/api/stats`
- **Supabase Dashboard**: https://supabase.com/dashboard/project/tvjalxxsyryjphkforjv
- **Vercel Dashboard**: https://vercel.com/lotd/curiouskelly
- **Resend Dashboard**: https://resend.com/emails

---

## Daily Emails Not Sending

**Symptoms**: Users report not receiving daily email

**Check**:
```bash
# Manual trigger
curl -H "Authorization: Bearer $CRON_SECRET" https://curiouskelly.com/api/cron/daily-lesson
```

**Common causes**:
1. CRON_SECRET mismatch → Check Vercel env vars
2. RESEND_API_KEY invalid → Check Resend dashboard for errors
3. No subscribed users → Check stats endpoint

**Fix**: Redeploy or manually trigger cron

---

## User Can't Sign Up

**Symptoms**: Error on signup, no confirmation email

**Check**:
1. Supabase Auth logs: Dashboard → Authentication → Logs
2. Email provider: Resend dashboard → Emails

**Common causes**:
1. Supabase Auth rate limited → Wait or increase limits
2. Email domain blocked → Check spam, verify DNS
3. SMTP misconfigured → Check Supabase Auth → Email Templates

---

## High Database Load

**Symptoms**: Slow API responses, timeouts

**Check**: Supabase Dashboard → Database → Query Performance

**Quick fixes**:
1. Check for missing indexes
2. Look for N+1 queries in logs
3. Enable connection pooling if not already

---

## Unsubscribe Not Working

**Symptoms**: User clicks unsubscribe, still receives emails

**Check**: 
```sql
SELECT email, email_daily_lesson, email_unsubscribed_at 
FROM users WHERE email = 'user@example.com';
```

**Fix**: Manually update if needed:
```sql
UPDATE users 
SET email_daily_lesson = false, email_unsubscribed_at = NOW() 
WHERE email = 'user@example.com';
```

---

## Lesson Page 404

**Symptoms**: /day/X returns 404

**Check**:
```sql
SELECT day_number, title FROM lessons WHERE day_number = X;
```

**Fix**: If lesson missing, check import ran successfully

---

## Welcome Email Not Sent

**Symptoms**: New user signs up, no welcome email

**Check**:
1. Supabase webhook configured? Dashboard → Auth → Hooks
2. Webhook secret matches Vercel env var?
3. Check Vercel function logs for errors

**Fix**: Manually send welcome or fix webhook config

---

## Streak Not Updating

**Symptoms**: User completes lesson, streak stays same

**Check**:
```sql
SELECT current_streak, last_lesson_at FROM users WHERE email = 'user@example.com';
```

**Logic**: Streak only increases if last_lesson_at was yesterday

---

## Emergency Contacts

- **Email issues**: hello@curiouskelly.com (auto-forwards)
- **Technical escalation**: Check Vercel/Supabase status pages

---

## Key Environment Variables

| Variable | Purpose |
|----------|---------|
| PUBLIC_SUPABASE_URL | Database URL |
| SUPABASE_SERVICE_ROLE_KEY | Admin database access |
| RESEND_API_KEY | Email sending |
| CRON_SECRET | Protects cron endpoints |
| SUPABASE_WEBHOOK_SECRET | Validates auth webhooks |

---

## Cron Schedule (UTC)

| Time | Endpoint | Purpose |
|------|----------|---------|
| 12:00 | /api/cron/daily-lesson | Daily emails (7am EST) |
| 08:00 | /api/cron/birthday-emails | Birthday emails (3am EST) |
| 18:00 | /api/cron/gentle-return | Re-engagement (1pm EST) |

