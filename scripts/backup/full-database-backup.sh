#!/bin/bash
###############################################################################
# Curious Kelly - Full Database Backup Script
# 
# Purpose: Creates a complete PostgreSQL dump of Supabase database
# Schedule: Daily at 3 AM UTC via GitHub Actions
# Storage: Cloudflare R2 (S3-compatible)
#
# Usage: ./full-database-backup.sh
#
# Required Environment Variables:
#   SUPABASE_DB_URL - Full PostgreSQL connection string
#   CLOUDFLARE_R2_ENDPOINT - R2 endpoint URL
#   CLOUDFLARE_R2_ACCESS_KEY - R2 access key
#   CLOUDFLARE_R2_SECRET_KEY - R2 secret key
#   CLOUDFLARE_R2_BUCKET - R2 bucket name
###############################################################################

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DATE=$(date +%Y-%m-%d)
BACKUP_TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
BACKUP_DIR="./backups"
BACKUP_FILENAME="curious-kelly-${BACKUP_DATE}.sql"
BACKUP_FILENAME_GZ="curious-kelly-${BACKUP_DATE}.sql.gz"
SCHEMA_FILENAME="curious-kelly-${BACKUP_DATE}-schema.sql"
RETENTION_DAYS=30

# Logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Check required environment variables
check_env() {
    log "Checking environment variables..."
    
    local required_vars=(
        "SUPABASE_DB_URL"
        "CLOUDFLARE_R2_ENDPOINT"
        "CLOUDFLARE_R2_ACCESS_KEY"
        "CLOUDFLARE_R2_SECRET_KEY"
        "CLOUDFLARE_R2_BUCKET"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    log "✓ All required environment variables are set"
}

# Check if required tools are installed
check_dependencies() {
    log "Checking dependencies..."
    
    local required_tools=("pg_dump" "aws" "gzip")
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "$tool is not installed"
            exit 1
        fi
    done
    
    log "✓ All required tools are available"
}

# Create backup directory
prepare_backup_dir() {
    log "Preparing backup directory..."
    mkdir -p "$BACKUP_DIR"
    log "✓ Backup directory ready: $BACKUP_DIR"
}

# Perform full database backup
backup_database() {
    log "Starting full database backup..."
    
    # Test connection first
    log "Testing database connection..."
    if ! psql "$SUPABASE_DB_URL" -c "SELECT 1;" > /dev/null 2>&1; then
        error "Cannot connect to database. Check SUPABASE_DB_URL."
        error "Connection string format: postgresql://postgres:PASSWORD@HOST:5432/postgres"
        exit 1
    fi
    log "✓ Database connection successful"
    
    # Full backup with data
    log "Running pg_dump..."
    if pg_dump "$SUPABASE_DB_URL" \
        --no-owner \
        --no-acl \
        --clean \
        --if-exists \
        --verbose \
        --file="$BACKUP_DIR/$BACKUP_FILENAME" 2>&1 | grep -v "NOTICE"; then
        log "✓ Database backup completed: $BACKUP_FILENAME"
    else
        local exit_code=$?
        error "Database backup failed with exit code: $exit_code"
        error "Check database connection and credentials"
        exit 1
    fi
    
    # Schema-only backup for reference
    if pg_dump "$SUPABASE_DB_URL" \
        --no-owner \
        --no-acl \
        --schema-only \
        --file="$BACKUP_DIR/$SCHEMA_FILENAME" 2>&1 | grep -v "NOTICE"; then
        log "✓ Schema backup completed: $SCHEMA_FILENAME"
    else
        warn "Schema backup failed (non-critical)"
    fi
}

# Compress backup files
compress_backup() {
    log "Compressing backup..."
    
    if gzip -f "$BACKUP_DIR/$BACKUP_FILENAME"; then
        local original_size=$(stat -f%z "$BACKUP_DIR/$BACKUP_FILENAME_GZ" 2>/dev/null || stat -c%s "$BACKUP_DIR/$BACKUP_FILENAME_GZ")
        log "✓ Backup compressed: $(numfmt --to=iec $original_size 2>/dev/null || echo $original_size bytes)"
    else
        error "Compression failed"
        exit 1
    fi
}

# Configure AWS CLI for Cloudflare R2
configure_s3_client() {
    log "Configuring S3 client for Cloudflare R2..."
    
    if [[ -z "${CLOUDFLARE_R2_ACCESS_KEY:-}" ]] || [[ -z "${CLOUDFLARE_R2_SECRET_KEY:-}" ]]; then
        error "R2 credentials not set. Check CLOUDFLARE_R2_ACCESS_KEY and CLOUDFLARE_R2_SECRET_KEY"
        exit 1
    fi
    
    export AWS_ACCESS_KEY_ID="$CLOUDFLARE_R2_ACCESS_KEY"
    export AWS_SECRET_ACCESS_KEY="$CLOUDFLARE_R2_SECRET_KEY"
    export AWS_DEFAULT_REGION="auto"
    
    log "✓ S3 client configured"
}

# Upload backup to Cloudflare R2
upload_to_r2() {
    log "Uploading backup to Cloudflare R2..."
    
    # Verify AWS CLI is configured
    if ! command -v aws &> /dev/null; then
        error "AWS CLI is not installed or not in PATH"
        exit 1
    fi
    
    # Credentials should already be set by configure_s3_client()
    # But verify they're available for AWS CLI
    if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]] || [[ -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
        error "AWS credentials not set. configure_s3_client() should have set these."
        exit 1
    fi
    
    local s3_path="s3://${CLOUDFLARE_R2_BUCKET}/daily/${BACKUP_FILENAME_GZ}"
    
    log "Uploading to: $s3_path"
    log "Endpoint: $CLOUDFLARE_R2_ENDPOINT"
    log "Bucket: $CLOUDFLARE_R2_BUCKET"
    
    # Upload with explicit error handling
    local upload_output
    upload_output=$(aws s3 cp \
        "$BACKUP_DIR/$BACKUP_FILENAME_GZ" \
        "$s3_path" \
        --endpoint-url "$CLOUDFLARE_R2_ENDPOINT" \
        --no-progress 2>&1)
    local upload_exit_code=$?
    
    if [[ $upload_exit_code -eq 0 ]]; then
        log "✓ Backup uploaded to R2: $s3_path"
        log "Upload output: $upload_output"
    else
        error "Upload to R2 failed with exit code: $upload_exit_code"
        error "AWS CLI output: $upload_output"
        error "Check R2 credentials and endpoint URL"
        exit 1
    fi
    
    # Upload schema file (uncompressed for easy viewing)
    local schema_s3_path="s3://${CLOUDFLARE_R2_BUCKET}/daily/${SCHEMA_FILENAME}"
    
    log "Uploading schema to: $schema_s3_path"
    
    local schema_upload_output
    schema_upload_output=$(aws s3 cp \
        "$BACKUP_DIR/$SCHEMA_FILENAME" \
        "$schema_s3_path" \
        --endpoint-url "$CLOUDFLARE_R2_ENDPOINT" \
        --no-progress 2>&1)
    local schema_upload_exit_code=$?
    
    if [[ $schema_upload_exit_code -eq 0 ]]; then
        log "✓ Schema uploaded to R2: $schema_s3_path"
    else
        warn "Schema upload failed (non-critical) - exit code: $schema_upload_exit_code"
        warn "AWS CLI output: $schema_upload_output"
    fi
    
    # Verify upload succeeded by listing R2 bucket
    log "Verifying upload by listing R2 bucket contents..."
    local verify_output
    verify_output=$(aws s3 ls "s3://${CLOUDFLARE_R2_BUCKET}/daily/" --endpoint-url "$CLOUDFLARE_R2_ENDPOINT" 2>&1)
    local verify_exit_code=$?
    
    if [[ $verify_exit_code -eq 0 ]]; then
        log "✓ R2 bucket listing successful:"
        echo "$verify_output" | while read -r line; do
            log "  $line"
        done
        
        # Check if our file is in the list
        if echo "$verify_output" | grep -q "$BACKUP_FILENAME_GZ"; then
            log "✓ Confirmed: Backup file is in R2 bucket"
        else
            error "⚠ WARNING: Backup file not found in R2 bucket listing!"
            error "This may indicate an upload failure despite success code"
        fi
    else
        warn "Could not verify upload (listing failed): $verify_output"
    fi
}

# Verify backup integrity
verify_backup() {
    log "Verifying backup integrity..."
    
    # Check if file exists and is not empty
    if [[ -f "$BACKUP_DIR/$BACKUP_FILENAME_GZ" ]] && [[ -s "$BACKUP_DIR/$BACKUP_FILENAME_GZ" ]]; then
        local file_size=$(stat -f%z "$BACKUP_DIR/$BACKUP_FILENAME_GZ" 2>/dev/null || stat -c%s "$BACKUP_DIR/$BACKUP_FILENAME_GZ")
        
        # Sanity check: backup should be at least 1MB
        if [[ $file_size -lt 1048576 ]]; then
            warn "Backup file is suspiciously small (< 1MB): $(numfmt --to=iec $file_size 2>/dev/null || echo $file_size bytes)"
        else
            log "✓ Backup integrity check passed: $(numfmt --to=iec $file_size 2>/dev/null || echo $file_size bytes)"
        fi
        
        # Test gzip integrity
        if gzip -t "$BACKUP_DIR/$BACKUP_FILENAME_GZ" 2>&1; then
            log "✓ Backup compression is valid"
        else
            error "Backup file is corrupted"
            exit 1
        fi
    else
        error "Backup file does not exist or is empty"
        exit 1
    fi
}

# Clean up old local backups
cleanup_local() {
    log "Cleaning up local backups older than $RETENTION_DAYS days..."
    
    find "$BACKUP_DIR" -name "curious-kelly-*.sql.gz" -type f -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR" -name "curious-kelly-*-schema.sql" -type f -mtime +$RETENTION_DAYS -delete
    
    log "✓ Local cleanup completed"
}

# Clean up old R2 backups
cleanup_r2() {
    log "Cleaning up R2 backups older than $RETENTION_DAYS days..."
    
    local cutoff_date=$(date -u -d "$RETENTION_DAYS days ago" +%Y-%m-%d 2>/dev/null || date -u -v-${RETENTION_DAYS}d +%Y-%m-%d)
    
    # List and delete old backups
    aws s3 ls \
        "s3://${CLOUDFLARE_R2_BUCKET}/daily/" \
        --endpoint-url "$CLOUDFLARE_R2_ENDPOINT" \
        | while read -r line; do
            file_date=$(echo "$line" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' | head -1)
            file_name=$(echo "$line" | awk '{print $4}')
            
            if [[ "$file_date" < "$cutoff_date" ]]; then
                log "Deleting old backup: $file_name"
                aws s3 rm \
                    "s3://${CLOUDFLARE_R2_BUCKET}/daily/$file_name" \
                    --endpoint-url "$CLOUDFLARE_R2_ENDPOINT"
            fi
        done
    
    log "✓ R2 cleanup completed"
}

# Generate backup report
generate_report() {
    log "Generating backup report..."
    
    cat <<EOF

═══════════════════════════════════════════════════════
  CURIOUS KELLY DATABASE BACKUP REPORT
═══════════════════════════════════════════════════════
Date:           $BACKUP_TIMESTAMP
Backup File:    $BACKUP_FILENAME_GZ
Schema File:    $SCHEMA_FILENAME
Storage:        Cloudflare R2
Bucket:         $CLOUDFLARE_R2_BUCKET
Status:         ✅ SUCCESS

File Sizes:
$(ls -lh "$BACKUP_DIR/$BACKUP_FILENAME_GZ" 2>/dev/null | awk '{print "  Full Backup: " $5}')
$(ls -lh "$BACKUP_DIR/$SCHEMA_FILENAME" 2>/dev/null | awk '{print "  Schema:      " $5}')

Retention:      $RETENTION_DAYS days
Next Backup:    $(date -u -d '1 day' +%Y-%m-%d' 03:00 UTC' 2>/dev/null || date -u -v+1d +%Y-%m-%d' 03:00 UTC')
═══════════════════════════════════════════════════════

EOF
}

# Send notification (stub - implement email/Slack as needed)
send_notification() {
    local status=$1
    local message=$2
    
    if [[ "$status" == "success" ]]; then
        log "✓ Backup completed successfully"
        log "$message"
    else
        error "Backup failed: $message"
    fi
    
    # TODO: Implement email/Slack notification
    # Example: curl -X POST $SLACK_WEBHOOK_URL -d "{'text': '$message'}"
}

# Main execution
main() {
    log "═══════════════════════════════════════════════════════"
    log "  Starting Curious Kelly Database Backup"
    log "═══════════════════════════════════════════════════════"
    
    check_env
    check_dependencies
    prepare_backup_dir
    
    backup_database
    compress_backup
    verify_backup
    
    configure_s3_client
    upload_to_r2
    
    cleanup_local
    cleanup_r2
    
    generate_report
    send_notification "success" "Database backup completed successfully"
    
    log "═══════════════════════════════════════════════════════"
    log "✅ BACKUP COMPLETED SUCCESSFULLY"
    log "═══════════════════════════════════════════════════════"
}

# Run main function
main "$@"





