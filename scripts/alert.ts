#!/usr/bin/env npx tsx
/**
 * Alerting utility for generation engine.
 * Sends alerts via:
 * 1. Console log (always)
 * 2. Write to logs/alerts/YYYY-MM-DD.json
 * 3. Optional: webhook to Slack/Discord/email (configurable)
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';

type AlertType = 
  | 'GENERATION_STARTED'
  | 'GENERATION_COMPLETE'
  | 'GENERATION_FAILED'
  | 'DAILY_LIMIT_REACHED'
  | 'PHASE_COMPLETE'
  | 'PHASE_FAILED'
  | 'REVIEW_NEEDED'
  | 'DAY_APPROVED'
  | 'ROLLBACK_TRIGGERED'
  | 'SYSTEM_ERROR';

interface AlertPayload {
  day?: number;
  days?: number[];
  phase?: string;
  error?: string;
  cost?: number;
  spent?: number;
  limit?: number;
  service?: string;
  details?: Record<string, unknown>;
}

interface AlertEntry {
  timestamp: string;
  type: AlertType;
  payload: AlertPayload;
  sent_to: string[];
}

const ALERTS_DIR = path.join(process.cwd(), 'logs', 'alerts');
const WEBHOOK_URL = process.env.ALERT_WEBHOOK_URL;

function getEmoji(type: AlertType): string {
  const emojis: Record<AlertType, string> = {
    GENERATION_STARTED: '🚀',
    GENERATION_COMPLETE: '✅',
    GENERATION_FAILED: '❌',
    DAILY_LIMIT_REACHED: '🛑',
    PHASE_COMPLETE: '✔️',
    PHASE_FAILED: '⚠️',
    REVIEW_NEEDED: '👀',
    DAY_APPROVED: '👍',
    ROLLBACK_TRIGGERED: '⏪',
    SYSTEM_ERROR: '🔥',
  };
  return emojis[type] || '📢';
}

function formatMessage(type: AlertType, payload: AlertPayload): string {
  const emoji = getEmoji(type);
  
  switch (type) {
    case 'GENERATION_STARTED':
      return `${emoji} Generation started for days: ${payload.days?.join(', ') || payload.day}`;
    case 'GENERATION_COMPLETE':
      return `${emoji} Generation complete! Days: ${payload.days?.join(', ')}. Cost: $${payload.cost?.toFixed(2)}`;
    case 'GENERATION_FAILED':
      return `${emoji} Generation FAILED for day ${payload.day}: ${payload.error}`;
    case 'DAILY_LIMIT_REACHED':
      return `${emoji} DAILY LIMIT REACHED! Spent: $${payload.spent?.toFixed(2)} / Limit: $${payload.limit}`;
    case 'PHASE_COMPLETE':
      return `${emoji} Day ${payload.day} phase ${payload.phase} complete`;
    case 'PHASE_FAILED':
      return `${emoji} Day ${payload.day} phase ${payload.phase} FAILED: ${payload.error}`;
    case 'REVIEW_NEEDED':
      return `${emoji} Day ${payload.day} ready for review`;
    case 'DAY_APPROVED':
      return `${emoji} Day ${payload.day} APPROVED and published`;
    case 'ROLLBACK_TRIGGERED':
      return `${emoji} Day ${payload.day} rolled back`;
    case 'SYSTEM_ERROR':
      return `${emoji} SYSTEM ERROR: ${payload.error}`;
    default:
      return `${emoji} ${type}: ${JSON.stringify(payload)}`;
  }
}

async function sendWebhook(message: string, type: AlertType): Promise<boolean> {
  if (!WEBHOOK_URL) return false;
  
  try {
    const response = await fetch(WEBHOOK_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: message,
        type,
        timestamp: new Date().toISOString(),
      }),
    });
    return response.ok;
  } catch (error) {
    console.error('Webhook failed:', error);
    return false;
  }
}

function writeToLog(entry: AlertEntry): void {
  const today = new Date().toISOString().split('T')[0];
  const logPath = path.join(ALERTS_DIR, `${today}.json`);
  
  // Ensure directory exists
  if (!fs.existsSync(ALERTS_DIR)) {
    fs.mkdirSync(ALERTS_DIR, { recursive: true });
  }
  
  // Read existing or create new
  let entries: AlertEntry[] = [];
  if (fs.existsSync(logPath)) {
    try {
      entries = JSON.parse(fs.readFileSync(logPath, 'utf-8'));
    } catch {
      entries = [];
    }
  }
  
  entries.push(entry);
  fs.writeFileSync(logPath, JSON.stringify(entries, null, 2));
}

export async function alert(type: AlertType, payload: AlertPayload = {}): Promise<void> {
  const timestamp = new Date().toISOString();
  const message = formatMessage(type, payload);
  const sentTo: string[] = ['console'];
  
  // Always log to console
  console.log(`[${timestamp}] ${message}`);
  
  // Write to log file
  try {
    writeToLog({ timestamp, type, payload, sent_to: sentTo });
    sentTo.push('file');
  } catch (error) {
    console.error('Failed to write alert to log:', error);
  }
  
  // Send webhook if configured
  if (WEBHOOK_URL) {
    const webhookSent = await sendWebhook(message, type);
    if (webhookSent) {
      sentTo.push('webhook');
    }
  }
}

// CLI usage: npx tsx scripts/alert.ts TYPE [JSON_PAYLOAD]
// tsconfig in this repo targets CommonJS, so `import.meta` is not available here.
// Node-style main-module detection works reliably for scripts.
// eslint-disable-next-line @typescript-eslint/no-var-requires
const isMainModule = require.main === module;

if (isMainModule) {
  const [, , type, payloadJson] = process.argv;
  
  if (!type) {
    console.log(`
Usage: npx tsx scripts/alert.ts <TYPE> [JSON_PAYLOAD]

Types:
  GENERATION_STARTED    - Generation job started
  GENERATION_COMPLETE   - Generation finished successfully
  GENERATION_FAILED     - Generation encountered error
  DAILY_LIMIT_REACHED   - Budget/quota exceeded
  PHASE_COMPLETE        - Single phase finished
  PHASE_FAILED          - Single phase failed
  REVIEW_NEEDED         - Content ready for human review
  DAY_APPROVED          - Day marked as approved
  ROLLBACK_TRIGGERED    - Content rolled back
  SYSTEM_ERROR          - Critical system error

Examples:
  npx tsx scripts/alert.ts GENERATION_COMPLETE '{"days": [354, 355], "cost": 12.40}'
  npx tsx scripts/alert.ts GENERATION_FAILED '{"day": 354, "error": "HeyGen timeout"}'
  npx tsx scripts/alert.ts DAILY_LIMIT_REACHED '{"spent": 48.50, "limit": 50}'
`);
    process.exit(1);
  }
  
  let payload: AlertPayload = {};
  if (payloadJson) {
    try {
      payload = JSON.parse(payloadJson);
    } catch {
      console.error('Invalid JSON payload');
      process.exit(1);
    }
  }
  
  alert(type as AlertType, payload).then(() => {
    console.log('Alert sent.');
  });
}




