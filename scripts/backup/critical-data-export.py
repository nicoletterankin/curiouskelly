#!/usr/bin/env python3
"""
Curious Kelly - Critical Data Export Script

Purpose: Exports critical user and lesson data to CSV/JSON for fast recovery
Schedule: Every 6 hours via GitHub Actions
Storage: Cloudflare R2

This script exports only the most critical data in portable formats:
- users (CSV)
- lessons (JSON)
- user_progress (CSV)

These exports enable quick data analysis and partial recovery scenarios.
"""

import os
import sys
import json
import csv
import gzip
import boto3
from datetime import datetime
from typing import Dict, List, Any
import psycopg2
from psycopg2.extras import RealDictCursor

# Configuration
BACKUP_DATE = datetime.utcnow().strftime('%Y-%m-%d')
BACKUP_TIMESTAMP = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
BACKUP_DIR = './backups/critical-data'
RETENTION_DAYS = 30

# Tables to export
EXPORT_TABLES = {
    'users': {
        'filename': f'users-{BACKUP_DATE}.csv',
        'columns': ['id', 'email', 'name', 'subscription_tier', 'subscription_status', 
                   'current_day', 'streak_days', 'last_lesson_at', 'created_at'],
        'format': 'csv'
    },
    'lessons': {
        'filename': f'lessons-{BACKUP_DATE}.json',
        'columns': ['id', 'day_number', 'title', 'subtitle', 'content', 'audio_url', 
                   'duration_seconds', 'difficulty', 'tags', 'is_published'],
        'format': 'json'
    },
    'user_progress': {
        'filename': f'user-progress-{BACKUP_DATE}.csv',
        'columns': ['id', 'user_id', 'lesson_id', 'completed', 'progress_percent', 
                   'time_spent_seconds', 'completed_at', 'started_at'],
        'format': 'csv'
    }
}


def log(message: str):
    """Print log message with timestamp"""
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")


def error(message: str):
    """Print error message and exit"""
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def check_env():
    """Check required environment variables"""
    log("Checking environment variables...")
    
    required_vars = [
        'SUPABASE_DB_URL',
        'CLOUDFLARE_R2_ENDPOINT',
        'CLOUDFLARE_R2_ACCESS_KEY',
        'CLOUDFLARE_R2_SECRET_KEY',
        'CLOUDFLARE_R2_BUCKET'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        error(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    log("✓ All required environment variables are set")


def prepare_backup_dir():
    """Create backup directory if it doesn't exist"""
    log("Preparing backup directory...")
    os.makedirs(BACKUP_DIR, exist_ok=True)
    log(f"✓ Backup directory ready: {BACKUP_DIR}")


def connect_to_database():
    """Connect to Supabase PostgreSQL database"""
    log("Connecting to database...")
    
    try:
        conn = psycopg2.connect(os.getenv('SUPABASE_DB_URL'))
        log("✓ Database connection established")
        return conn
    except Exception as e:
        error(f"Failed to connect to database: {e}")


def export_table_to_csv(conn, table_name: str, config: Dict[str, Any]) -> str:
    """Export table data to CSV file"""
    log(f"Exporting {table_name} to CSV...")
    
    filename = config['filename']
    columns = config['columns']
    filepath = os.path.join(BACKUP_DIR, filename)
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Build query with specified columns
            columns_sql = ', '.join(columns)
            query = f"SELECT {columns_sql} FROM public.{table_name} ORDER BY created_at DESC"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            if not rows:
                log(f"⚠ No data found in {table_name}")
                return filepath
            
            # Write to CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()
                
                for row in rows:
                    # Convert datetime objects to ISO format strings
                    row_dict = {}
                    for key, value in row.items():
                        if isinstance(value, datetime):
                            row_dict[key] = value.isoformat()
                        else:
                            row_dict[key] = value
                    writer.writerow(row_dict)
            
            log(f"✓ Exported {len(rows)} rows from {table_name} to {filename}")
            return filepath
            
    except Exception as e:
        error(f"Failed to export {table_name}: {e}")


def export_table_to_json(conn, table_name: str, config: Dict[str, Any]) -> str:
    """Export table data to JSON file"""
    log(f"Exporting {table_name} to JSON...")
    
    filename = config['filename']
    columns = config['columns']
    filepath = os.path.join(BACKUP_DIR, filename)
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Build query with specified columns
            columns_sql = ', '.join(columns)
            query = f"SELECT {columns_sql} FROM public.{table_name} ORDER BY day_number"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            if not rows:
                log(f"⚠ No data found in {table_name}")
                return filepath
            
            # Convert to JSON-serializable format
            json_data = []
            for row in rows:
                row_dict = {}
                for key, value in row.items():
                    if isinstance(value, datetime):
                        row_dict[key] = value.isoformat()
                    elif isinstance(value, (list, dict)):
                        row_dict[key] = value  # Already JSON-compatible
                    else:
                        row_dict[key] = value
                json_data.append(row_dict)
            
            # Write to JSON file
            with open(filepath, 'w', encoding='utf-8') as jsonfile:
                json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)
            
            log(f"✓ Exported {len(rows)} rows from {table_name} to {filename}")
            return filepath
            
    except Exception as e:
        error(f"Failed to export {table_name}: {e}")


def compress_file(filepath: str) -> str:
    """Compress file with gzip"""
    log(f"Compressing {os.path.basename(filepath)}...")
    
    gz_filepath = f"{filepath}.gz"
    
    try:
        with open(filepath, 'rb') as f_in:
            with gzip.open(gz_filepath, 'wb') as f_out:
                f_out.writelines(f_in)
        
        # Remove original file
        os.remove(filepath)
        
        file_size = os.path.getsize(gz_filepath)
        log(f"✓ Compressed to {os.path.basename(gz_filepath)} ({file_size:,} bytes)")
        return gz_filepath
        
    except Exception as e:
        error(f"Failed to compress {filepath}: {e}")


def upload_to_r2(filepath: str):
    """Upload file to Cloudflare R2"""
    log(f"Uploading {os.path.basename(filepath)} to R2...")
    
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('CLOUDFLARE_R2_ENDPOINT'),
            aws_access_key_id=os.getenv('CLOUDFLARE_R2_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('CLOUDFLARE_R2_SECRET_KEY'),
            region_name='auto'
        )
        
        bucket = os.getenv('CLOUDFLARE_R2_BUCKET')
        key = f"critical-data/{os.path.basename(filepath)}"
        
        s3_client.upload_file(filepath, bucket, key)
        
        log(f"✓ Uploaded to s3://{bucket}/{key}")
        
    except Exception as e:
        error(f"Failed to upload to R2: {e}")


def cleanup_old_files():
    """Remove files older than retention period"""
    log(f"Cleaning up files older than {RETENTION_DAYS} days...")
    
    cutoff_timestamp = datetime.utcnow().timestamp() - (RETENTION_DAYS * 86400)
    removed_count = 0
    
    if os.path.exists(BACKUP_DIR):
        for filename in os.listdir(BACKUP_DIR):
            filepath = os.path.join(BACKUP_DIR, filename)
            
            if os.path.isfile(filepath):
                file_timestamp = os.path.getmtime(filepath)
                
                if file_timestamp < cutoff_timestamp:
                    os.remove(filepath)
                    removed_count += 1
    
    log(f"✓ Removed {removed_count} old files")


def generate_report(exported_files: List[str]):
    """Generate export report"""
    log("Generating export report...")
    
    print("\n" + "=" * 63)
    print("  CURIOUS KELLY CRITICAL DATA EXPORT REPORT")
    print("=" * 63)
    print(f"Date:           {BACKUP_TIMESTAMP}")
    print(f"Storage:        Cloudflare R2")
    print(f"Bucket:         {os.getenv('CLOUDFLARE_R2_BUCKET')}")
    print(f"Status:         ✅ SUCCESS")
    print(f"\nExported Files:")
    
    for filepath in exported_files:
        file_size = os.path.getsize(filepath)
        print(f"  - {os.path.basename(filepath)} ({file_size:,} bytes)")
    
    print(f"\nRetention:      {RETENTION_DAYS} days")
    print(f"Next Export:    {datetime.utcnow().strftime('%Y-%m-%d %H:00')} UTC")
    print("=" * 63)
    print()


def main():
    """Main execution"""
    log("=" * 63)
    log("  Starting Curious Kelly Critical Data Export")
    log("=" * 63)
    
    check_env()
    prepare_backup_dir()
    
    conn = connect_to_database()
    exported_files = []
    
    try:
        # Export each table
        for table_name, config in EXPORT_TABLES.items():
            if config['format'] == 'csv':
                filepath = export_table_to_csv(conn, table_name, config)
            elif config['format'] == 'json':
                filepath = export_table_to_json(conn, table_name, config)
            else:
                error(f"Unknown format: {config['format']}")
            
            # Compress and upload
            gz_filepath = compress_file(filepath)
            upload_to_r2(gz_filepath)
            exported_files.append(gz_filepath)
        
        cleanup_old_files()
        generate_report(exported_files)
        
        log("=" * 63)
        log("✅ CRITICAL DATA EXPORT COMPLETED SUCCESSFULLY")
        log("=" * 63)
        
    except Exception as e:
        error(f"Export failed: {e}")
    
    finally:
        conn.close()


if __name__ == '__main__':
    main()



























