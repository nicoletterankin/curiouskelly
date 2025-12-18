# 🔒 AUTONOMOUS OPERATIONS — LOCKED

> **Lock Date:** December 18, 2025  
> **Status:** PRODUCTION  
> **Owner:** nicoletterankin@gmail.com  
> **Philosophy:** Complete software. Zero day-to-day intervention required.

---

## 🚫 DO NOT MODIFY WITHOUT EXPLICIT APPROVAL

This system is locked. Any changes require:
1. Written justification
2. Impact analysis
3. Rollback plan
4. Testing in staging first

---

## 📧 NOTIFICATION SCHEDULE

| Cron | Time | File | Purpose | Recipient |
|------|------|------|---------|-----------|
| Happy Digest | 8 PM daily | `api/cron/happy-learner-digest.ts` | Celebrations | nicoletterankin@gmail.com |
| Escalation | 9 AM daily | `api/cron/escalation-check.ts` | Issues needing attention | nicoletterankin@gmail.com |
| Weekly | Sunday 6 PM | `api/cron/weekly-digest.ts` | Full summary | nicoletterankin@gmail.com |
| Streak Check | 11 PM daily | `api/cron/streak-check.ts` | Log milestones | (internal) |
| HeyGen Monitor | Every 4 hrs | `api/cron/heygen-monitor.ts` | Video status | (internal) |

---

## 🤖 AUTO-MODERATION RULES (DATABASE TRIGGERS)

| Rule | Threshold | Action | Reversible |
|------|-----------|--------|------------|
| Trusted User | 5+ approved comments | Auto-approve future comments | Yes (manual) |
| Popular Comment | 10+ upvotes | Auto-feature | Yes (manual) |
| Community Accept | 20+ upvotes on suggestion | Auto-accept | No |
| Community Reject | 10+ downvotes on suggestion | Auto-decline | No |

---

## 🚨 ESCALATION RULES

Only escalate to founder when:
1. ⏰ Moderation pending > 48 hours
2. 📈 Suggestion has 5+ upvotes but no response > 48 hours
3. 💳 Payment failure unresolved
4. 🐛 Bug report open

**Everything else:** Community self-resolves via voting.

---

## 🗄️ DATABASE TABLES (LOCKED SCHEMA)

### `founder_notifications`
```sql
id UUID PRIMARY KEY
type TEXT CHECK (type IN ('milestone', 'escalation', 'happy_digest', 'escalation_digest', 'weekly_digest'))
data JSONB
sent_at TIMESTAMP WITH TIME ZONE
created_at TIMESTAMP WITH TIME ZONE
```

### `happy_learner_events`
```sql
id UUID PRIMARY KEY
type TEXT CHECK (type IN ('first_lesson', 'streak_7', 'streak_30', 'streak_100', 'streak_365', 'completed_track', 'helpful_comment', 'first_comment'))
user_id UUID REFERENCES auth.users(id)
detail TEXT
created_at TIMESTAMP WITH TIME ZONE
```

### `lesson_completions`
```sql
id UUID PRIMARY KEY
user_id UUID REFERENCES auth.users(id)
lesson_year INTEGER
lesson_day INTEGER
completed_at TIMESTAMP WITH TIME ZONE
UNIQUE(user_id, lesson_year, lesson_day)
```

### `payment_events`
```sql
id UUID PRIMARY KEY
user_id UUID REFERENCES auth.users(id)
stripe_customer_id TEXT
event_type TEXT
amount_cents INTEGER
currency TEXT DEFAULT 'usd'
resolved BOOLEAN DEFAULT false
metadata JSONB
created_at TIMESTAMP WITH TIME ZONE
```

### `heygen_performance_logs`
```sql
id UUID PRIMARY KEY
checked_at TIMESTAMP WITH TIME ZONE
completed_count INTEGER
pending_count INTEGER
failed_count INTEGER
sample_data JSONB
created_at TIMESTAMP WITH TIME ZONE
```

---

## 🔐 SECURITY REQUIREMENTS

1. **Cron Authentication:** All crons verify `CRON_SECRET` bearer token
2. **Service Role Only:** Database writes use `SUPABASE_SERVICE_ROLE_KEY`
3. **Email Sender:** Only `hello@curiouskelly.com` authorized
4. **Recipient Lock:** Only `nicoletterankin@gmail.com` receives alerts
5. **Rate Limiting:** Max 1 email per cron execution
6. **Idempotency:** Duplicate events prevented by unique constraints

---

## ✅ VERIFICATION CHECKLIST

Before considering this system operational:

- [ ] All cron jobs registered in Vercel
- [ ] `CRON_SECRET` environment variable set
- [ ] `SENDGRID_API_KEY` environment variable set
- [ ] `SUPABASE_SERVICE_ROLE_KEY` environment variable set
- [ ] All database tables exist
- [ ] All triggers active
- [ ] Health endpoint responding
- [ ] Test email received

---

## 🔄 RECOVERY PROCEDURES

### If emails stop arriving:
1. Check Vercel Functions logs
2. Verify SendGrid API status
3. Check `founder_notifications` table for recent entries
4. Manually trigger: `curl https://curiouskelly.com/api/cron/weekly-digest`

### If escalations aren't triggering:
1. Check `phase_comments` for pending > 48h
2. Check `curriculum_suggestions` for open with upvotes
3. Verify trigger functions exist in Supabase

### If auto-moderation fails:
1. Check Supabase trigger logs
2. Verify `auto_moderate_comment` function exists
3. Test manually with INSERT

---

## 📊 MONITORING

- **Health Endpoint:** `/api/health`
- **Cron Logs:** Vercel Dashboard → Functions
- **Email Logs:** `founder_notifications` table
- **Happy Events:** `happy_learner_events` table

---

## 🔒 LOCK SIGNATURE

This document represents the complete specification for autonomous operations.
No modifications without explicit founder approval.

```
LOCKED: 2025-12-18T02:30:00Z
HASH: sha256(founder-alerts + auto-moderation + escalation-rules)
```
