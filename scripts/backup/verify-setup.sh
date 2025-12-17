#!/bin/bash
###############################################################################
# Curious Kelly - Backup Setup Verification Script
#
# Purpose: Verify that all backup components are properly configured
# Usage: ./verify-setup.sh
###############################################################################

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

error() {
    echo -e "${RED}[✗]${NC} $1"
    ERRORS=$((ERRORS + 1))
}

warn() {
    echo -e "${YELLOW}[!]${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

header() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  $1"
    echo "═══════════════════════════════════════════════════════"
}

# Check if command exists
check_command() {
    local cmd=$1
    local install_msg=$2
    
    if command -v "$cmd" &> /dev/null; then
        success "$cmd is installed"
        return 0
    else
        error "$cmd is NOT installed"
        warn "Install with: $install_msg"
        return 1
    fi
}

# Check environment variable
check_env_var() {
    local var=$1
    local description=$2
    
    if [[ -n "${!var:-}" ]]; then
        success "$var is set"
        return 0
    else
        error "$var is NOT set"
        warn "Description: $description"
        return 1
    fi
}

# Check file exists
check_file() {
    local file=$1
    local description=$2
    
    if [[ -f "$file" ]]; then
        success "$file exists"
        return 0
    else
        error "$file does NOT exist"
        warn "Description: $description"
        return 1
    fi
}

# Main verification
main() {
    header "Curious Kelly Backup Setup Verification"
    
    # Check dependencies
    header "1. Checking Required Tools"
    check_command "pg_dump" "sudo apt-get install postgresql-client (Ubuntu) or brew install postgresql (macOS)"
    check_command "aws" "sudo apt-get install awscli (Ubuntu) or brew install awscli (macOS)"
    check_command "gzip" "Usually pre-installed"
    check_command "python3" "sudo apt-get install python3 (Ubuntu) or brew install python3 (macOS)"
    check_command "pip" "sudo apt-get install python3-pip (Ubuntu) or python3 -m ensurepip (macOS)"
    
    # Check Python packages
    header "2. Checking Python Dependencies"
    if python3 -c "import psycopg2" 2>/dev/null; then
        success "psycopg2 is installed"
    else
        error "psycopg2 is NOT installed"
        warn "Install with: pip install psycopg2-binary"
    fi
    
    if python3 -c "import boto3" 2>/dev/null; then
        success "boto3 is installed"
    else
        error "boto3 is NOT installed"
        warn "Install with: pip install boto3"
    fi
    
    # Check environment variables
    header "3. Checking Environment Variables"
    check_env_var "SUPABASE_DB_URL" "PostgreSQL connection string for Supabase"
    check_env_var "CLOUDFLARE_R2_ENDPOINT" "R2 endpoint URL (e.g., https://[account-id].r2.cloudflarestorage.com)"
    check_env_var "CLOUDFLARE_R2_ACCESS_KEY" "R2 API access key"
    check_env_var "CLOUDFLARE_R2_SECRET_KEY" "R2 API secret key"
    check_env_var "CLOUDFLARE_R2_BUCKET" "R2 bucket name (e.g., curious-kelly-backups)"
    
    # Check files
    header "4. Checking Backup Scripts"
    check_file "scripts/backup/full-database-backup.sh" "Full database backup script"
    check_file "scripts/backup/critical-data-export.py" "Critical data export script"
    check_file ".github/workflows/database-backup.yml" "GitHub Actions workflow"
    check_file "scripts/backup/requirements.txt" "Python dependencies file"
    
    # Check script permissions
    header "5. Checking Script Permissions"
    if [[ -f "scripts/backup/full-database-backup.sh" ]]; then
        if [[ -x "scripts/backup/full-database-backup.sh" ]]; then
            success "full-database-backup.sh is executable"
        else
            warn "full-database-backup.sh is NOT executable"
            warn "Fix with: chmod +x scripts/backup/full-database-backup.sh"
        fi
    fi
    
    if [[ -f "scripts/backup/critical-data-export.py" ]]; then
        if [[ -x "scripts/backup/critical-data-export.py" ]]; then
            success "critical-data-export.py is executable"
        else
            warn "critical-data-export.py is NOT executable"
            warn "Fix with: chmod +x scripts/backup/critical-data-export.py"
        fi
    fi
    
    # Test database connection (if env vars set)
    if [[ -n "${SUPABASE_DB_URL:-}" ]]; then
        header "6. Testing Database Connection"
        if psql "$SUPABASE_DB_URL" -c "SELECT version();" &>/dev/null; then
            success "Database connection successful"
        else
            error "Database connection FAILED"
            warn "Check SUPABASE_DB_URL and network connectivity"
        fi
    else
        warn "Skipping database connection test (SUPABASE_DB_URL not set)"
    fi
    
    # Test R2 connection (if env vars set)
    if [[ -n "${CLOUDFLARE_R2_ENDPOINT:-}" ]] && [[ -n "${CLOUDFLARE_R2_ACCESS_KEY:-}" ]]; then
        header "7. Testing R2 Connection"
        
        export AWS_ACCESS_KEY_ID="$CLOUDFLARE_R2_ACCESS_KEY"
        export AWS_SECRET_ACCESS_KEY="$CLOUDFLARE_R2_SECRET_KEY"
        
        if aws s3 ls "s3://${CLOUDFLARE_R2_BUCKET}/" --endpoint-url "$CLOUDFLARE_R2_ENDPOINT" &>/dev/null; then
            success "R2 connection successful"
        else
            error "R2 connection FAILED"
            warn "Check R2 credentials and bucket name"
        fi
    else
        warn "Skipping R2 connection test (credentials not set)"
    fi
    
    # Check documentation
    header "8. Checking Documentation"
    check_file "docs/backend/DATABASE_BACKUP_PLAN.md" "Comprehensive backup plan"
    check_file "docs/backend/DATABASE_RESTORE_PROCEDURES.md" "Restore procedures"
    check_file "docs/backend/BACKUP_SETUP_QUICKSTART.md" "Quick start guide"
    
    # Summary
    header "Verification Summary"
    
    if [[ $ERRORS -eq 0 ]] && [[ $WARNINGS -eq 0 ]]; then
        echo -e "${GREEN}✅ ALL CHECKS PASSED!${NC}"
        echo ""
        echo "Your backup system is ready to use."
        echo ""
        echo "Next steps:"
        echo "1. Run a manual test backup: ./scripts/backup/full-database-backup.sh"
        echo "2. Verify backup appears in R2 storage"
        echo "3. Review docs/backend/BACKUP_SETUP_QUICKSTART.md for automation setup"
        echo ""
        return 0
    elif [[ $ERRORS -eq 0 ]]; then
        echo -e "${YELLOW}⚠ PASSED WITH WARNINGS${NC}"
        echo ""
        echo "Warnings found: $WARNINGS"
        echo "Your backup system should work, but please review warnings above."
        echo ""
        return 0
    else
        echo -e "${RED}❌ VERIFICATION FAILED${NC}"
        echo ""
        echo "Errors found: $ERRORS"
        echo "Warnings found: $WARNINGS"
        echo ""
        echo "Please fix the errors above before proceeding."
        echo "Refer to docs/backend/BACKUP_SETUP_QUICKSTART.md for setup instructions."
        echo ""
        return 1
    fi
}

# Run verification
main "$@"




























