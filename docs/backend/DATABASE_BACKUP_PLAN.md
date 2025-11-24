# Supabase Database Backup Strategy
## Senior Engineer Plan - Production Grade

**Project:** Curious Kelly  
**Database:** Supabase PostgreSQL (Project ID: tvjalxxsyryjphkforjv)  
**Last Updated:** November 24, 2025  
**Owner:** Platform Engineering

---

## Executive Summary

This document defines a comprehensive, production-grade backup strategy for the Curious Kelly Supabase database. The strategy includes automated daily backups, point-in-time recovery capabilities, disaster recovery procedures, and compliance with data retention policies.

### Critical Requirements
- **RPO (Recovery Point Objective):** 1 hour max data loss
- **RTO (Recovery Time Objective):** 4 hours max downtime
- **Retention:** 30 days point-in-time, 1 year archived
- **Compliance:** GDPR/CCPA ready (PII protection)

---

## 1. Backup Architecture

### 1.1 Multi-Layer Backup Strategy

```
┌─────────────────────────────────────────────────────┐
│ Layer 1: Supabase Native Backups (Automated)       │
│ - Daily automatic backups                           │
│ - 7-day retention on Pro plan                       │
│ - Point-in-time recovery (PITR)                     │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Layer 2: Custom pg_dump Backups (Scheduled)        │
│ - Daily full dumps                                   │
│ - Stored in Cloudflare R2 / AWS S3                  │
│ - 30-day retention                                   │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Layer 3: Long-term Archives (Monthly)               │
│ - Monthly snapshots                                  │
│ - Stored in cold storage                             │
│ - 1-year retention                                   │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Layer 4: Critical Data Exports (Weekly)            │
│ - User data exports (CSV)                            │
│ - Lesson content exports (JSON)                      │
│ - Stored separately for fast recovery               │
└─────────────────────────────────────────────────────┘
```

### 1.2 What Gets Backed Up

**Public Schema Tables:**
- `users` - User profiles and subscription data
- `lessons` - 365-day curriculum content
- `user_progress` - Progress tracking and streaks
- `affiliates` - Affiliate program data
- `referrals` - Referral tracking
- `affiliate_applications` - Applications
- `enterprise_inquiries` - Sales leads
- `newsletter_subscribers` - Email list
- `analytics_events` - User behavior data

**Auth Schema Tables:**
- `auth.users` - Authentication records
- `auth.identities` - Social login connections

**Database Objects:**
- Functions, triggers, policies (RLS)
- Indexes and constraints
- Sequences and extensions

---

## 2. Implementation Plan

### 2.1 Enable Supabase Native Backups

**Status:** ✅ Already enabled on Supabase Pro plan

Supabase provides:
- Automated daily backups
- Point-in-time recovery (PITR) for last 7 days
- Accessible via Dashboard → Database → Backups

**Action Required:**
1. Verify Pro plan subscription
2. Test restore from Supabase dashboard
3. Document restore procedure

### 2.2 Custom Automated Backups

**Implementation:** GitHub Actions + Cloud Storage

Create automated backup system that runs:
- **Daily at 3 AM UTC** - Full database dump
- **Every 6 hours** - Incremental data exports (critical tables only)
- **Weekly on Sundays** - Full schema + data validation test

### 2.3 Storage Strategy

**Primary Backup Storage: Cloudflare R2**
- Cost-effective ($0.015/GB/month)
- No egress fees (free retrieval)
- S3-compatible API
- 99.999999999% durability

**Alternative: AWS S3**
- More expensive but battle-tested
- Glacier for long-term archives
- Cross-region replication available

**Backup Naming Convention:**
```
backups/
  daily/
    curious-kelly-{YYYY-MM-DD}.sql
    curious-kelly-{YYYY-MM-DD}.sql.gz
    curious-kelly-{YYYY-MM-DD}-schema.sql
  weekly/
    curious-kelly-week-{YYYY-WW}.sql.gz
    curious-kelly-week-{YYYY-WW}-data.tar
  monthly/
    curious-kelly-{YYYY-MM}.sql.gz
    curious-kelly-{YYYY-MM}-archive.tar.gz
  critical-data/
    users-{YYYY-MM-DD}.csv
    lessons-{YYYY-MM-DD}.json
    user-progress-{YYYY-MM-DD}.csv
```

---

## 3. Backup Scripts

### 3.1 Database Connection Configuration

Required environment variables:
```bash
SUPABASE_DB_URL=postgresql://postgres:[PASSWORD]@db.tvjalxxsyryjphkforjv.supabase.co:5432/postgres
SUPABASE_PROJECT_REF=tvjalxxsyryjphkforjv
CLOUDFLARE_R2_ENDPOINT=https://[account-id].r2.cloudflarestorage.com
CLOUDFLARE_R2_ACCESS_KEY=[key]
CLOUDFLARE_R2_SECRET_KEY=[secret]
CLOUDFLARE_R2_BUCKET=curious-kelly-backups
```

### 3.2 Backup Execution Scripts

See accompanying files:
- `scripts/backup/full-database-backup.sh` - Complete pg_dump
- `scripts/backup/incremental-backup.sh` - Data-only exports
- `scripts/backup/critical-data-export.py` - User/lesson exports
- `.github/workflows/database-backup.yml` - GitHub Actions automation

### 3.3 Restore Procedures

See: `docs/backend/DATABASE_RESTORE_PROCEDURES.md`

---

## 4. Security & Compliance

### 4.1 Backup Encryption

**At Rest:**
- All backups encrypted with AES-256
- Encryption keys stored in GitHub Secrets / 1Password
- Never commit keys to repository

**In Transit:**
- TLS 1.3 for all database connections
- HTTPS for S3/R2 uploads

### 4.2 Access Control

**Backup Access Restricted To:**
- GitHub Actions service account (automated backups)
- Platform owner (emergency restore only)
- No developer access to production backups

**Audit Trail:**
- All backup/restore operations logged
- Logs retained for 1 year

### 4.3 PII Handling

**GDPR/CCPA Compliance:**
- Backups contain user emails and subscription data
- User data deletion requests must purge backups
- Implement "right to be forgotten" procedure

**Retention Policy:**
- Regular backups: 30 days
- Monthly archives: 1 year
- After retention period: Secure deletion (overwrite + verify)

---

## 5. Monitoring & Alerting

### 5.1 Backup Health Checks

**Daily Verification:**
- ✅ Backup file created
- ✅ Backup uploaded to R2/S3
- ✅ Backup file size > 1MB (sanity check)
- ✅ Backup contains expected table count
- ✅ Backup restoration test (weekly)

**Alerts Sent To:**
- Email: hello@curiouskelly.com
- Slack: #platform-alerts (if configured)
- PagerDuty: Critical failures only

### 5.2 Failure Scenarios & Responses

| Failure | Detection | Response | SLA |
|---------|-----------|----------|-----|
| Backup job failed | GitHub Actions alert | Retry immediately, escalate if 2nd failure | 1 hour |
| Upload to R2 failed | Upload verification check | Retry with exponential backoff | 2 hours |
| Backup file corrupted | Weekly restore test | Use previous backup, investigate corruption | 24 hours |
| Supabase PITR unavailable | Dashboard check | Rely on custom backups, contact Supabase | 4 hours |

---

## 6. Disaster Recovery Procedures

### 6.1 Recovery Scenarios

**Scenario A: Accidental Data Deletion (< 7 days ago)**
- Use Supabase PITR to restore to specific timestamp
- Estimated recovery time: 30 minutes
- Data loss: 0 minutes to recovery point

**Scenario B: Database Corruption (> 7 days ago)**
- Restore from custom pg_dump backup
- Estimated recovery time: 2-4 hours
- Data loss: Up to 24 hours (last daily backup)

**Scenario C: Complete Supabase Project Loss**
- Provision new Supabase project
- Restore from most recent backup
- Reconfigure authentication providers
- Update connection strings in frontend
- Estimated recovery time: 4-8 hours

**Scenario D: Ransomware / Security Breach**
- Isolate compromised systems
- Restore from last known-good backup (pre-breach)
- Force password resets for all users
- Incident response: 8-24 hours

### 6.2 Restoration Testing

**Quarterly Restore Drill:**
- Provision test Supabase project
- Restore from most recent backup
- Verify data integrity (row counts, user authentication)
- Test critical application flows
- Document any issues

**Last Successful Test:** TBD (schedule first test)

---

## 7. Cost Analysis

### 7.1 Estimated Monthly Costs

| Service | Usage | Cost |
|---------|-------|------|
| Supabase Pro Plan | Includes native backups | $25/mo |
| Cloudflare R2 Storage | 10GB daily backups × 30 days = 300GB | $4.50/mo |
| GitHub Actions | ~1 hour/month compute | Free (included) |
| **Total** | | **$29.50/mo** |

### 7.2 Storage Growth Projections

| Timeframe | Est. Database Size | Backup Storage | Monthly Cost |
|-----------|-------------------|----------------|--------------|
| Launch (Dec 2025) | 500MB | 15GB | $0.23 |
| 6 months | 5GB | 150GB | $2.25 |
| 1 year | 20GB | 600GB | $9.00 |
| 2 years | 80GB | 2.4TB | $36.00 |

---

## 8. Rollout Plan

### Phase 1: Immediate (This Week)
- [ ] Create Cloudflare R2 bucket `curious-kelly-backups`
- [ ] Generate R2 API credentials
- [ ] Store credentials in GitHub Secrets
- [ ] Test manual backup/restore locally
- [ ] Document manual procedures

### Phase 2: Automation (Next Week)
- [ ] Create backup scripts (shell + Python)
- [ ] Set up GitHub Actions workflow
- [ ] Configure daily backup schedule
- [ ] Test automated backup job
- [ ] Set up monitoring alerts

### Phase 3: Validation (Week 3)
- [ ] Run first weekly restoration test
- [ ] Verify backup integrity
- [ ] Measure restore time (RTO)
- [ ] Document any issues
- [ ] Update procedures as needed

### Phase 4: Production (Ongoing)
- [ ] Monitor backup health daily
- [ ] Review storage costs monthly
- [ ] Run quarterly DR drills
- [ ] Update retention policies as needed

---

## 9. Responsibilities

| Task | Owner | Frequency |
|------|-------|-----------|
| Monitor backup job success | Platform Owner | Daily |
| Review backup storage costs | Platform Owner | Monthly |
| Test database restore | Platform Owner | Weekly (automated) |
| Conduct DR drill | Platform Owner | Quarterly |
| Update backup scripts | Platform Owner | As needed |
| Review retention policies | Platform Owner | Annually |

---

## 10. References

**External Documentation:**
- [Supabase Backup Docs](https://supabase.com/docs/guides/platform/backups)
- [PostgreSQL pg_dump](https://www.postgresql.org/docs/current/app-pgdump.html)
- [Cloudflare R2 Docs](https://developers.cloudflare.com/r2/)
- [GDPR Right to Erasure](https://gdpr-info.eu/art-17-gdpr/)

**Internal Documentation:**
- `docs/backend/SUPABASE_SCHEMA.md` - Database schema
- `docs/backend/DATABASE_RESTORE_PROCEDURES.md` - Restore playbook
- `scripts/backup/` - Backup automation scripts
- `CLAUDE.md` - Secrets management rules

---

## Approval & Sign-off

**Plan Status:** ✅ Approved for Implementation  
**Approved By:** Platform Owner  
**Date:** November 24, 2025  
**Next Review:** December 24, 2025

---

**END OF BACKUP PLAN**

