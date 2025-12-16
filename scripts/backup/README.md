# Database Backup Scripts

Automated backup system for Curious Kelly Supabase database.

## Files

- **`full-database-backup.sh`** - Complete PostgreSQL dump with schema and data
- **`critical-data-export.py`** - Export critical tables to CSV/JSON
- **`README.md`** - This file

## Quick Start

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install postgresql-client awscli

# macOS
brew install postgresql awscli

# Python dependencies
pip install psycopg2-binary boto3
```

### Environment Variables

Create `.env` file (never commit this!):

```bash
SUPABASE_DB_URL=postgresql://postgres:[PASSWORD]@db.tvjalxxsyryjphkforjv.supabase.co:5432/postgres
CLOUDFLARE_R2_ENDPOINT=https://[account-id].r2.cloudflarestorage.com
CLOUDFLARE_R2_ACCESS_KEY=[your-access-key]
CLOUDFLARE_R2_SECRET_KEY=[your-secret-key]
CLOUDFLARE_R2_BUCKET=curious-kelly-backups
```

Load environment variables:

```bash
source .env  # or
export $(cat .env | xargs)
```

### Run Backups Manually

```bash
# Full database backup
chmod +x full-database-backup.sh
./full-database-backup.sh

# Critical data export
chmod +x critical-data-export.py
python critical-data-export.py
```

## Automated Schedule (GitHub Actions)

Configured in `.github/workflows/database-backup.yml`:

- **Daily at 3:00 AM UTC** - Full database backup
- **Every 6 hours** - Critical data export
- **Sundays at 4:00 AM UTC** - Restore test

## Backup Outputs

### Full Database Backup

**Location:** `s3://curious-kelly-backups/daily/`

**Files:**
- `curious-kelly-YYYY-MM-DD.sql.gz` - Compressed full backup
- `curious-kelly-YYYY-MM-DD-schema.sql` - Schema only (reference)

**Size:** Typically 5-50 MB compressed

### Critical Data Export

**Location:** `s3://curious-kelly-backups/critical-data/`

**Files:**
- `users-YYYY-MM-DD.csv.gz` - User profiles and subscriptions
- `lessons-YYYY-MM-DD.json.gz` - All lesson content
- `user-progress-YYYY-MM-DD.csv.gz` - Progress tracking data

**Size:** Typically 1-10 MB compressed

## Restoring from Backup

See: `docs/backend/DATABASE_RESTORE_PROCEDURES.md`

Quick restore:

```bash
# Download backup
aws s3 cp \
  s3://curious-kelly-backups/daily/curious-kelly-2025-11-24.sql.gz \
  ./backup.sql.gz \
  --endpoint-url $CLOUDFLARE_R2_ENDPOINT

# Decompress
gunzip backup.sql.gz

# Restore (CAUTION: This will overwrite existing data)
psql $SUPABASE_DB_URL < backup.sql
```

## Monitoring

### Check Backup Success

**GitHub Actions:**
- Go to repository → Actions → Database Backup
- View recent workflow runs
- Green checkmark = success
- Red X = failure (investigate immediately)

**R2 Storage:**
- Log in to Cloudflare Dashboard
- Navigate to R2 → `curious-kelly-backups`
- Verify new files appear daily

### Alerts

- Failures automatically create GitHub Issues
- Email notifications sent to configured address
- Check Actions tab for detailed logs

## Retention Policy

- **Daily backups:** 30 days
- **Critical data:** 30 days
- **Monthly archives:** 1 year (manual process)

Old backups are automatically deleted by scripts.

## Storage Costs

**Cloudflare R2:**
- Storage: $0.015/GB/month
- No egress fees (free downloads)

**Estimated costs:**
- 1-10 GB database: $0.15-$1.50/month
- 10-100 GB database: $1.50-$15/month
- 100+ GB database: $15+/month

## Security

- **Never commit credentials** to git
- Store secrets in GitHub Secrets or secure vault
- Use read-only credentials where possible
- Encrypt backups (already handled by R2)
- Limit access to backup storage

## Troubleshooting

### "pg_dump: command not found"

Install PostgreSQL client:

```bash
# Ubuntu/Debian
sudo apt-get install postgresql-client

# macOS
brew install postgresql
```

### "aws: command not found"

Install AWS CLI:

```bash
# Ubuntu/Debian
sudo apt-get install awscli

# macOS
brew install awscli
```

### "connection refused"

Check database URL and network:

```bash
# Test connection
psql $SUPABASE_DB_URL -c "SELECT version();"
```

### "access denied" to R2

Verify credentials:

```bash
aws s3 ls s3://curious-kelly-backups/ \
  --endpoint-url $CLOUDFLARE_R2_ENDPOINT
```

## Development

### Testing Changes

Always test backup scripts locally before pushing:

```bash
# Test full backup
./full-database-backup.sh

# Verify output
ls -lh backups/

# Test upload (won't affect production if bucket name differs)
```

### Adding New Tables

Edit `critical-data-export.py`:

```python
EXPORT_TABLES = {
    'your_new_table': {
        'filename': f'your-table-{BACKUP_DATE}.csv',
        'columns': ['id', 'name', 'created_at'],
        'format': 'csv'
    }
}
```

## Documentation

- **Full Backup Plan:** `docs/backend/DATABASE_BACKUP_PLAN.md`
- **Restore Procedures:** `docs/backend/DATABASE_RESTORE_PROCEDURES.md`
- **Quick Start Guide:** `docs/backend/BACKUP_SETUP_QUICKSTART.md`
- **Schema Documentation:** `docs/backend/SUPABASE_SCHEMA.md`

## Support

**Issues:** Create GitHub Issue with labels `database` and `backup`

**Questions:** Check documentation first, then ask team

---

**Last Updated:** November 24, 2025
























