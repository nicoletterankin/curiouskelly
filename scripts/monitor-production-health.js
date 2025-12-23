#!/usr/bin/env node
/**
 * Enterprise-Grade Production Health Monitor
 * Continuous monitoring of conversational narration functionality
 * Hard-coded checks, no assumptions
 */

import ProductionVerifier from './verify-production-conversational.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const MONITOR_INTERVAL = 5 * 60 * 1000; // 5 minutes
const LOG_FILE = path.join(__dirname, '../logs/production-health.log');
const ALERT_THRESHOLD = 3; // Alert after 3 consecutive failures

class ProductionHealthMonitor {
  constructor() {
    this.verifier = new ProductionVerifier();
    this.consecutiveFailures = 0;
    this.lastStatus = null;
    this.monitoring = false;
    this.ensureLogDirectory();
  }

  ensureLogDirectory() {
    const logDir = path.dirname(LOG_FILE);
    if (!fs.existsSync(logDir)) {
      fs.mkdirSync(logDir, { recursive: true });
    }
  }

  log(message, level = 'INFO') {
    const timestamp = new Date().toISOString();
    const logEntry = `[${timestamp}] [${level}] ${message}\n`;
    fs.appendFileSync(LOG_FILE, logEntry);
    console.log(`[${timestamp}] ${message}`);
  }

  async checkHealth() {
    try {
      this.log('Starting health check...');
      const results = await this.verifier.verifyProduction();
      const report = this.verifier.generateReport();

      if (results.overall === 'PASS') {
        this.consecutiveFailures = 0;
        this.lastStatus = 'HEALTHY';
        this.log(`✅ Health check PASSED - ${report.summary.passedChecks}/${report.summary.totalChecks} checks passed`, 'SUCCESS');
        return true;
      } else {
        this.consecutiveFailures++;
        this.lastStatus = 'UNHEALTHY';
        this.log(`❌ Health check FAILED - ${report.summary.failedChecks} checks failed`, 'ERROR');
        
        if (this.consecutiveFailures >= ALERT_THRESHOLD) {
          this.alert(report);
        }
        return false;
      }
    } catch (error) {
      this.consecutiveFailures++;
      this.lastStatus = 'ERROR';
      this.log(`💥 Health check ERROR: ${error.message}`, 'ERROR');
      return false;
    }
  }

  alert(report) {
    const alertMessage = `
🚨 PRODUCTION ALERT 🚨
Consecutive failures: ${this.consecutiveFailures}
Critical failures: ${report.summary.criticalFailures}
Failed checks: ${report.summary.failedChecks}
Timestamp: ${new Date().toISOString()}
    `.trim();

    this.log(alertMessage, 'ALERT');
    
    // In production, this would send to monitoring service (PagerDuty, etc.)
    // For now, write to alert file
    const alertFile = path.join(__dirname, '../logs/production-alert.log');
    fs.appendFileSync(alertFile, alertMessage + '\n\n');
  }

  start() {
    if (this.monitoring) {
      this.log('Monitor already running');
      return;
    }

    this.monitoring = true;
    this.log('🚀 Starting production health monitor');
    this.log(`Check interval: ${MONITOR_INTERVAL / 1000} seconds`);

    // Initial check
    this.checkHealth();

    // Periodic checks
    this.interval = setInterval(() => {
      this.checkHealth();
    }, MONITOR_INTERVAL);
  }

  stop() {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
    this.monitoring = false;
    this.log('⏹️  Production health monitor stopped');
  }

  getStatus() {
    return {
      monitoring: this.monitoring,
      lastStatus: this.lastStatus,
      consecutiveFailures: this.consecutiveFailures,
      lastCheck: this.verifier.results.timestamp
    };
  }
}

// CLI interface
if (import.meta.url === `file://${process.argv[1]}`) {
  const monitor = new ProductionHealthMonitor();
  
  // Handle graceful shutdown
  process.on('SIGINT', () => {
    console.log('\nShutting down monitor...');
    monitor.stop();
    process.exit(0);
  });

  process.on('SIGTERM', () => {
    console.log('\nShutting down monitor...');
    monitor.stop();
    process.exit(0);
  });

  // Start monitoring
  monitor.start();

  // Keep process alive
  process.stdin.resume();
}

export default ProductionHealthMonitor;

