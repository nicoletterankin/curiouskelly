# Database Restore Procedures
## Curious Kelly - Supabase PostgreSQL

**Last Updated:** November 24, 2025  
**Owner:** Platform Engineering

---

## Quick Reference

| Scenario | Method | Estimated Time | Data Loss |
|----------|--------|----------------|-----------|
| Recent deletion (< 7 days) | Supabase PITR | 30 minutes | 0 minutes |
| Older data loss (> 7 days) | Custom backup restore | 2-4 hours | Up to 24 hours |
| Complete project loss | Full rebuild from backup | 4-8 hours | Up to 24 hours |
| Security breach | Restore pre-breach backup | 8-24 hours | Varies |

---

## Method 1: Supabase Point-in-Time Recovery (PITR)

### When to Use
- Accidental data deletion or modification within the last 7 days
- Need to restore to a specific timestamp
- Fastest recovery method

### Requirements
- Supabase Pro plan (PITR included)
- Event occurred within 7-day window
- Know approximate timestamp of incident

### Steps

1. **Access Supabase Dashboard**
   ```
   URL: https://supabase.com/dashboard/project/tvjalxxsyryjphkforjv
   ```

2. **Navigate to Backups**
   - Click "Database" in left sidebar
   - Select "Backups" tab
   - Look for "Point in Time Recovery" section

3. **Select Recovery Point**
   - Choose date/time to restore to
   - Should be BEFORE the incident occurred
   - View available recovery points

4. **Initiate Restore**
   - Click "Restore" button
   - Confirm restoration timestamp
   - **WARNING:** This will overwrite current database

5. **Monitor Progress**
   - Restoration typically takes 10-30 minutes
   - Database will be read-only during restore
   - Users will experience downtime

6. **Verify Data**
   - Check that missing/corrupted data is restored
   - Verify user counts, lesson data, progress records
   - Test critical application flows

7. **Resume Operations**
   - Notify users of service restoration
   - Monitor for any issues

### Troubleshooting

**Problem:** PITR not available  
**Solution:** Check Supabase plan (requires Pro), contact Supabase support

**Problem:** Can't find specific timestamp  
**Solution:** Use custom backup restore (Method 2)

---

## Method 2: Restore from Custom pg_dump Backup

### When to Use
- Data loss occurred more than 7 days ago
- PITR not available or insufficient
- Need to restore entire database

### Requirements
- Access to R2/S3 backup storage
- AWS CLI or S3-compatible client
- PostgreSQL client tools (pg_restore)
- Database connection credentials

### Steps

#### Step 1: Identify Backup to Restore

```bash
# Configure AWS CLI for R2
export AWS_ACCESS_KEY_ID="your-r2-access-key"
export AWS_SECRET_ACCESS_KEY="your-r2-secret-key"
export R2_ENDPOINT="https://your-account.r2.cloudflarestorage.com"
export R2_BUCKET="curious-kelly-backups"

# List available backups
aws s3 ls s3://${R2_BUCKET}/daily/ \
  --endpoint-url ${R2_ENDPOINT}

# Output example:
# 2025-11-20 curious-kelly-2025-11-20.sql.gz
# 2025-11-21 curious-kelly-2025-11-21.sql.gz
# 2025-11-22 curious-kelly-2025-11-22.sql.gz
# 2025-11-23 curious-kelly-2025-11-23.sql.gz
# 2025-11-24 curious-kelly-2025-11-24.sql.gz
```

#### Step 2: Download Backup

```bash
# Choose the backup date you want to restore
BACKUP_DATE="2025-11-23"
BACKUP_FILE="curious-kelly-${BACKUP_DATE}.sql.gz"

# Download backup
aws s3 cp \
  s3://${R2_BUCKET}/daily/${BACKUP_FILE} \
  ./${BACKUP_FILE} \
  --endpoint-url ${R2_ENDPOINT}

# Verify download
ls -lh ${BACKUP_FILE}
```

#### Step 3: Decompress Backup

```bash
# Decompress gzip file
gunzip ${BACKUP_FILE}

# This creates: curious-kelly-2025-11-23.sql
SQL_FILE="curious-kelly-${BACKUP_DATE}.sql"

# Verify integrity
head -n 20 ${SQL_FILE}
```

#### Step 4: Prepare Target Database

**Option A: Restore to Existing Supabase Project (DESTRUCTIVE)**

⚠️ **WARNING:** This will DELETE all current data

```bash
# Connect to Supabase
export SUPABASE_DB_URL="postgresql://postgres:[PASSWORD]@db.tvjalxxsyryjphkforjv.supabase.co:5432/postgres"

# Drop all tables (careful!)
psql ${SUPABASE_DB_URL} <<EOF
DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO public;
EOF
```

**Option B: Restore to New Supabase Project (RECOMMENDED)**

1. Create new Supabase project at https://supabase.com/dashboard
2. Note new project connection string
3. Update `SUPABASE_DB_URL` with new credentials

#### Step 5: Restore Database

```bash
# Restore from backup
psql ${SUPABASE_DB_URL} < ${SQL_FILE}

# This may take 5-30 minutes depending on database size
# Watch for errors in output
```

#### Step 6: Verify Restoration

```bash
# Connect to database
psql ${SUPABASE_DB_URL}

# Check table counts
SELECT 
  schemaname,
  tablename,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

# Verify critical tables
SELECT COUNT(*) FROM public.users;
SELECT COUNT(*) FROM public.lessons;
SELECT COUNT(*) FROM public.user_progress;

# Check most recent records
SELECT id, email, created_at FROM public.users ORDER BY created_at DESC LIMIT 10;
```

#### Step 7: Update Application Configuration

If you restored to a NEW Supabase project:

```bash
# Update environment variables
NEW_SUPABASE_URL="https://newproject.supabase.co"
NEW_SUPABASE_ANON_KEY="your-new-anon-key"

# Update in:
# 1. Vercel/Cloudflare environment variables
# 2. Local .env file
# 3. CI/CD secrets (GitHub Actions)
```

#### Step 8: Test Application

1. Test user authentication
2. Test lesson loading
3. Verify user progress tracking
4. Check affiliate/referral systems
5. Validate RLS policies are working

---

## Method 3: Partial Data Recovery (Critical Data Only)

### When to Use
- Only need to recover specific users or lessons
- Full restore not necessary
- Want to merge old data with current data

### Steps

#### Step 1: Download Critical Data Export

```bash
# List available exports
aws s3 ls s3://${R2_BUCKET}/critical-data/ \
  --endpoint-url ${R2_ENDPOINT}

# Download user export
aws s3 cp \
  s3://${R2_BUCKET}/critical-data/users-2025-11-23.csv.gz \
  ./users-backup.csv.gz \
  --endpoint-url ${R2_ENDPOINT}

# Decompress
gunzip users-backup.csv.gz
```

#### Step 2: Import Specific Records

```python
# Python script to selectively restore users
import csv
import psycopg2

conn = psycopg2.connect(os.getenv('SUPABASE_DB_URL'))
cursor = conn.cursor()

with open('users-backup.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Only restore specific user (example)
        if row['email'] == 'user@example.com':
            cursor.execute("""
                INSERT INTO public.users (id, email, name, subscription_tier, current_day, streak_days)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                  email = EXCLUDED.email,
                  name = EXCLUDED.name,
                  subscription_tier = EXCLUDED.subscription_tier
            """, (row['id'], row['email'], row['name'], row['subscription_tier'], 
                  row['current_day'], row['streak_days']))

conn.commit()
cursor.close()
conn.close()
```

---

## Emergency Contacts & Resources

### Supabase Support
- Dashboard: https://supabase.com/dashboard/project/tvjalxxsyryjphkforjv
- Support: https://supabase.com/dashboard/support
- Status: https://status.supabase.com

### Internal Resources
- Platform Owner: (check team docs)
- Backup Storage: Cloudflare R2
- Monitoring: GitHub Actions

### Documentation
- Supabase Backups: https://supabase.com/docs/guides/platform/backups
- PostgreSQL Recovery: https://www.postgresql.org/docs/current/backup-dump.html

---

## Post-Recovery Checklist

After any restore operation:

- [ ] Verify all critical tables have data
- [ ] Test user authentication (login/signup)
- [ ] Test lesson loading and playback
- [ ] Verify user progress tracking
- [ ] Check affiliate/referral functionality
- [ ] Validate RLS policies are active
- [ ] Update team on recovery status
- [ ] Document incident in runbook
- [ ] Review backup procedures for improvements
- [ ] Schedule post-mortem meeting

---

## Common Issues & Solutions

### Issue: "psql: error: connection to server ... failed"

**Cause:** Invalid connection string or network issue

**Solution:**
```bash
# Verify connection string format
echo $SUPABASE_DB_URL

# Test connection
psql ${SUPABASE_DB_URL} -c "SELECT version();"
```

### Issue: "permission denied for schema public"

**Cause:** Insufficient database privileges

**Solution:**
```sql
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO public;
```

### Issue: Backup file is corrupted

**Cause:** Download error or bad backup

**Solution:**
- Try re-downloading backup
- Use previous day's backup
- Check backup integrity from GitHub Actions logs

### Issue: RLS policies not working after restore

**Cause:** Policies may not have been restored

**Solution:**
```bash
# Re-apply schema from repo
psql ${SUPABASE_DB_URL} < supabase-schema.sql
```

---

## Testing Your Recovery Plan

**Recommended:** Test restore procedures quarterly

1. Create test Supabase project
2. Download most recent backup
3. Practice full restore
4. Time the recovery process
5. Document any issues
6. Update procedures as needed

---

**END OF RESTORE PROCEDURES**

















