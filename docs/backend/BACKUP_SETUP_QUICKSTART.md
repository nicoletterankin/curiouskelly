# Database Backup Setup - Quick Start Guide
## Get Your Backups Running in 30 Minutes

**Last Updated:** November 24, 2025

---

## Prerequisites

- [ ] Supabase Pro plan (required for PITR)
- [ ] Cloudflare account
- [ ] GitHub repository access
- [ ] 30 minutes of your time

---

## Step 1: Create Cloudflare R2 Bucket (5 minutes)

1. **Log in to Cloudflare Dashboard**
   - Go to https://dash.cloudflare.com
   - Navigate to R2 Object Storage

2. **Create Bucket**
   - Click "Create bucket"
   - Name: `curious-kelly-backups`
   - Location: Automatic
   - Click "Create bucket"

3. **Generate API Credentials**
   - Go to "Manage R2 API Tokens"
   - Click "Create API Token"
   - Name: `backup-automation`
   - Permissions: "Object Read & Write"
   - Bucket: `curious-kelly-backups`
   - Click "Create API Token"

4. **Save Credentials**
   ```
   Access Key ID: [SAVE THIS]
   Secret Access Key: [SAVE THIS]
   Endpoint URL: https://[account-id].r2.cloudflarestorage.com
   ```

---

## Step 2: Get Supabase Database Connection String (2 minutes)

1. **Access Supabase Dashboard**
   - Go to https://supabase.com/dashboard/project/tvjalxxsyryjphkforjv
   - Click "Settings" → "Database"

2. **Copy Connection String**
   - Look for "Connection string"
   - Select "URI" format
   - Click "Copy"
   - Format: `postgresql://postgres:[PASSWORD]@db.tvjalxxsyryjphkforjv.supabase.co:5432/postgres`

3. **Save Connection String**
   - Store securely (1Password, etc.)
   - You'll need this for GitHub Secrets

---

## Step 3: Configure GitHub Secrets (5 minutes)

1. **Navigate to Repository Settings**
   - Go to your GitHub repository
   - Click "Settings" → "Secrets and variables" → "Actions"

2. **Add Secrets**
   Click "New repository secret" for each:

   | Secret Name | Value | Where to Get It |
   |-------------|-------|-----------------|
   | `SUPABASE_DB_URL` | `postgresql://postgres:...` | Step 2 |
   | `CLOUDFLARE_R2_ENDPOINT` | `https://[id].r2.cloudflarestorage.com` | Step 1 |
   | `CLOUDFLARE_R2_ACCESS_KEY` | Your R2 Access Key | Step 1 |
   | `CLOUDFLARE_R2_SECRET_KEY` | Your R2 Secret Key | Step 1 |
   | `CLOUDFLARE_R2_BUCKET` | `curious-kelly-backups` | Step 1 |

3. **Verify All Secrets Are Added**
   - You should have 5 secrets total
   - Double-check names match exactly

---

## Step 4: Enable GitHub Actions Workflow (2 minutes)

The workflow file is already created at `.github/workflows/database-backup.yml`

1. **Commit and Push**
   ```bash
   git add .github/workflows/database-backup.yml
   git commit -m "Add automated database backup workflow"
   git push
   ```

2. **Verify Workflow is Active**
   - Go to "Actions" tab in GitHub
   - Look for "Database Backup" workflow
   - Should show as enabled

---

## Step 5: Test Manual Backup (10 minutes)

### Option A: Test via GitHub Actions (Recommended)

1. **Trigger Manual Backup**
   - Go to "Actions" → "Database Backup"
   - Click "Run workflow"
   - Select `backup_type: full`
   - Click "Run workflow"

2. **Monitor Progress**
   - Watch the workflow execution
   - Should complete in 3-5 minutes
   - Check for any errors

3. **Verify Backup in R2**
   - Log in to Cloudflare Dashboard
   - Navigate to R2 → `curious-kelly-backups`
   - Look for `daily/curious-kelly-[DATE].sql.gz`
   - File should be > 1MB

### Option B: Test Locally

```bash
# Install dependencies (if not already installed)
sudo apt-get install postgresql-client awscli

# Set environment variables
export SUPABASE_DB_URL="postgresql://postgres:..."
export CLOUDFLARE_R2_ENDPOINT="https://..."
export CLOUDFLARE_R2_ACCESS_KEY="..."
export CLOUDFLARE_R2_SECRET_KEY="..."
export CLOUDFLARE_R2_BUCKET="curious-kelly-backups"

# Make script executable
chmod +x scripts/backup/full-database-backup.sh

# Run backup
./scripts/backup/full-database-backup.sh
```

---

## Step 6: Set Up Monitoring (5 minutes)

### Email Notifications

1. **Enable GitHub Notifications**
   - Go to your GitHub profile settings
   - Click "Notifications"
   - Enable "Actions" notifications
   - Add email: hello@curiouskelly.com

2. **Test Notification**
   - Workflows will automatically create GitHub Issues on failure
   - You'll receive email when issues are created

### Optional: Slack Integration

If you have a Slack workspace:

```yaml
# Add to .github/workflows/database-backup.yml after failure steps
- name: Notify Slack
  if: failure()
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK_URL }}
    payload: |
      {
        "text": "🚨 Database Backup Failed",
        "blocks": [
          {
            "type": "section",
            "text": {
              "type": "mrkdwn",
              "text": "*Database Backup Failed*\n\nPlease investigate immediately."
            }
          }
        ]
      }
```

---

## Step 7: Verify Automated Schedule (1 minute)

Your backups are now scheduled automatically:

| Backup Type | Schedule | Frequency |
|-------------|----------|-----------|
| Full Database Backup | Daily at 3:00 AM UTC | Once per day |
| Critical Data Export | Every 6 hours | 4 times per day |
| Restore Test | Sundays at 4:00 AM UTC | Once per week |

**No further action needed!** ✅

---

## Verification Checklist

Before considering setup complete:

- [ ] R2 bucket created and accessible
- [ ] All 5 GitHub Secrets configured
- [ ] Workflow file committed to repository
- [ ] Manual test backup completed successfully
- [ ] Backup file visible in R2 bucket
- [ ] Backup file size is reasonable (> 1MB)
- [ ] Email notifications configured
- [ ] Reviewed backup and restore documentation

---

## What Happens Next?

### Daily (Automated)
- Full database backup at 3:00 AM UTC
- Uploaded to R2 storage
- Old backups cleaned up (30-day retention)

### Every 6 Hours (Automated)
- Critical data exported (users, lessons, progress)
- Lightweight CSV/JSON files
- Fast recovery option

### Weekly (Automated)
- Restore test performed
- Backup integrity verified
- Report uploaded to GitHub

### Your Responsibilities
- Monitor backup success (check GitHub Actions occasionally)
- Review monthly storage costs (should be ~$5-10/month)
- Conduct quarterly full restore drill
- Keep credentials secure

---

## Troubleshooting

### Workflow fails with "permission denied"

**Cause:** GitHub Actions doesn't have permission to create issues

**Solution:**
1. Go to Settings → Actions → General
2. Scroll to "Workflow permissions"
3. Select "Read and write permissions"
4. Save

### Backup fails with "connection timeout"

**Cause:** Database URL is incorrect or network issue

**Solution:**
1. Verify `SUPABASE_DB_URL` in GitHub Secrets
2. Test connection locally: `psql $SUPABASE_DB_URL -c "SELECT 1;"`
3. Check Supabase project is not paused

### Upload to R2 fails

**Cause:** R2 credentials are incorrect

**Solution:**
1. Regenerate R2 API token in Cloudflare
2. Update GitHub Secrets with new credentials
3. Retry workflow

---

## Next Steps

1. **Read Full Documentation**
   - `docs/backend/DATABASE_BACKUP_PLAN.md` - Complete strategy
   - `docs/backend/DATABASE_RESTORE_PROCEDURES.md` - Recovery playbook

2. **Schedule Quarterly DR Drill**
   - Practice full database restore
   - Time the recovery process
   - Update procedures based on learnings

3. **Monitor Costs**
   - Review Cloudflare R2 billing monthly
   - Expected: $5-10/month initially
   - Scales with database growth

---

## Support

**Questions?** Check the docs:
- Full backup plan: `docs/backend/DATABASE_BACKUP_PLAN.md`
- Restore procedures: `docs/backend/DATABASE_RESTORE_PROCEDURES.md`

**Issues?** Create a GitHub Issue with label `database` and `backup`

---

**Congratulations!** 🎉

Your database is now protected with:
- ✅ Daily automated backups
- ✅ 30-day retention
- ✅ Automatic monitoring
- ✅ Weekly restore tests
- ✅ Fast recovery options

Sleep well knowing your data is safe! 😴

---

**END OF QUICK START GUIDE**






