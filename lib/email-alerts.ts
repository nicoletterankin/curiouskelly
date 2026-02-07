/**
 * Email Alert System for Kelly Pipeline
 * 
 * Sends alerts to nicoletterankin@gmail.com when:
 * - Jobs fail after 3 retries
 * - Eval gates fail repeatedly
 * - Critical pipeline errors occur
 */

const ALERT_EMAIL = 'nicoletterankin@gmail.com';
const FROM_EMAIL = 'hello@curiouskelly.com';

export interface AlertPayload {
  type: 'eval_failure' | 'job_failure' | 'pipeline_error' | 'daily_summary';
  subject: string;
  body: string;
  job_id?: string;
  day_of_year?: number;
  phase?: string;
  engine?: string;
  retries?: number;
  issues?: string[];
  timestamp?: string;
}

/**
 * Send email alert via multiple methods
 * Tries: 1) Resend API, 2) Supabase Edge Function, 3) Console log
 */
export async function sendAlert(payload: AlertPayload): Promise<boolean> {
  const fullPayload = {
    ...payload,
    timestamp: payload.timestamp || new Date().toISOString(),
  };
  
  // Format email body
  const emailBody = formatEmailBody(fullPayload);
  
  // Try Resend API first
  if (process.env.RESEND_API_KEY) {
    try {
      const success = await sendViaResend(payload.subject, emailBody);
      if (success) return true;
    } catch (e) {
      console.error('Resend failed:', e);
    }
  }
  
  // Try Supabase Edge Function
  if (process.env.PUBLIC_SUPABASE_URL) {
    try {
      const success = await sendViaSupabase(fullPayload);
      if (success) return true;
    } catch (e) {
      console.error('Supabase email failed:', e);
    }
  }
  
  // Fallback: Console log (for debugging)
  console.log('\n' + '═'.repeat(60));
  console.log('📧 EMAIL ALERT (console fallback)');
  console.log('═'.repeat(60));
  console.log(`To: ${ALERT_EMAIL}`);
  console.log(`Subject: ${payload.subject}`);
  console.log('─'.repeat(60));
  console.log(emailBody);
  console.log('═'.repeat(60) + '\n');
  
  return true; // Console log always "succeeds"
}

/**
 * Send via Resend API
 */
async function sendViaResend(subject: string, body: string): Promise<boolean> {
  const response = await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${process.env.RESEND_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      from: FROM_EMAIL,
      to: ALERT_EMAIL,
      subject,
      text: body,
    }),
  });
  
  return response.ok;
}

/**
 * Send via Supabase Edge Function
 */
async function sendViaSupabase(payload: AlertPayload): Promise<boolean> {
  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL;
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  
  if (!supabaseUrl || !serviceKey) return false;
  
  const response = await fetch(`${supabaseUrl}/functions/v1/send-alert`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${serviceKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      to: ALERT_EMAIL,
      subject: payload.subject,
      payload,
    }),
  });
  
  return response.ok;
}

/**
 * Format email body for readability
 */
function formatEmailBody(payload: AlertPayload): string {
  const lines: string[] = [];
  
  lines.push(`🚨 KELLY PIPELINE ALERT`);
  lines.push('');
  lines.push(`Type: ${payload.type.toUpperCase()}`);
  lines.push(`Time: ${payload.timestamp}`);
  lines.push('');
  
  if (payload.job_id) {
    lines.push(`Job ID: ${payload.job_id}`);
  }
  if (payload.day_of_year !== undefined) {
    lines.push(`Day: ${payload.day_of_year}`);
  }
  if (payload.phase) {
    lines.push(`Phase: ${payload.phase}`);
  }
  if (payload.engine) {
    lines.push(`Engine: ${payload.engine}`);
  }
  if (payload.retries !== undefined) {
    lines.push(`Retries: ${payload.retries}`);
  }
  
  lines.push('');
  lines.push('─'.repeat(40));
  lines.push('');
  lines.push(payload.body);
  
  if (payload.issues && payload.issues.length > 0) {
    lines.push('');
    lines.push('Issues:');
    payload.issues.forEach(issue => {
      lines.push(`  • ${issue}`);
    });
  }
  
  lines.push('');
  lines.push('─'.repeat(40));
  lines.push('');
  lines.push('This is an automated alert from the Kelly Pipeline.');
  lines.push('Reply to this email if you need human assistance.');
  
  return lines.join('\n');
}

// ============================================
// ALERT TEMPLATES
// ============================================

export function alertEvalFailure(
  jobId: string,
  dayOfYear: number,
  phase: string,
  retries: number,
  issues: string[]
): AlertPayload {
  return {
    type: 'eval_failure',
    subject: `[Kelly] Eval Failed: Day ${dayOfYear} ${phase} (${retries} retries)`,
    body: `Content evaluation failed after ${retries} automatic retries.\n\nHuman review required.`,
    job_id: jobId,
    day_of_year: dayOfYear,
    phase,
    retries,
    issues,
  };
}

export function alertJobFailure(
  jobId: string,
  dayOfYear: number,
  phase: string,
  engine: string,
  error: string
): AlertPayload {
  return {
    type: 'job_failure',
    subject: `[Kelly] Job Failed: Day ${dayOfYear} ${phase} on ${engine}`,
    body: `Video generation job failed.\n\nEngine: ${engine}\nError: ${error}`,
    job_id: jobId,
    day_of_year: dayOfYear,
    phase,
    engine,
    issues: [error],
  };
}

export function alertPipelineError(error: string, context?: Record<string, any>): AlertPayload {
  return {
    type: 'pipeline_error',
    subject: '[Kelly] Pipeline Error - Immediate Attention Required',
    body: `A critical pipeline error occurred.\n\nError: ${error}\n\nContext: ${JSON.stringify(context, null, 2)}`,
    issues: [error],
  };
}

export function alertDailySummary(
  dayOfYear: number,
  stats: {
    completed: number;
    failed: number;
    pending: number;
    blocked: number;
  }
): AlertPayload {
  const total = stats.completed + stats.failed + stats.pending + stats.blocked;
  const successRate = total > 0 ? ((stats.completed / total) * 100).toFixed(1) : '0';
  
  return {
    type: 'daily_summary',
    subject: `[Kelly] Day ${dayOfYear} Summary: ${stats.completed}/${total} complete (${successRate}%)`,
    body: `Daily generation summary for Day ${dayOfYear}:\n\n` +
      `  ✅ Completed: ${stats.completed}\n` +
      `  ❌ Failed: ${stats.failed}\n` +
      `  ⏳ Pending: ${stats.pending}\n` +
      `  🚫 Blocked: ${stats.blocked}\n` +
      `\nSuccess Rate: ${successRate}%`,
    day_of_year: dayOfYear,
  };
}

// ============================================
// CONVENIENCE FUNCTIONS
// ============================================

export async function notifyEvalFailure(
  jobId: string,
  dayOfYear: number,
  phase: string,
  retries: number,
  issues: string[]
): Promise<boolean> {
  const payload = alertEvalFailure(jobId, dayOfYear, phase, retries, issues);
  return sendAlert(payload);
}

export async function notifyJobFailure(
  jobId: string,
  dayOfYear: number,
  phase: string,
  engine: string,
  error: string
): Promise<boolean> {
  const payload = alertJobFailure(jobId, dayOfYear, phase, engine, error);
  return sendAlert(payload);
}

export async function notifyPipelineError(
  error: string,
  context?: Record<string, any>
): Promise<boolean> {
  const payload = alertPipelineError(error, context);
  return sendAlert(payload);
}

export async function notifyDailySummary(
  dayOfYear: number,
  stats: { completed: number; failed: number; pending: number; blocked: number }
): Promise<boolean> {
  const payload = alertDailySummary(dayOfYear, stats);
  return sendAlert(payload);
}
