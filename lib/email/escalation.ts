/**
 * Kelly's Escalation Engine
 * 
 * Handles routing critical emails to nicoletterankin@gmail.com
 * and sending digest summaries of items needing attention.
 */

import { EmailClassification } from './classifier';

const RESEND_API_URL = 'https://api.resend.com/emails';
const ESCALATION_EMAIL = process.env.ESCALATION_EMAIL || 'nicoletterankin@gmail.com';

export interface EscalationContext {
  threadId: string;
  originalEmail: {
    from: string;
    fromName?: string;
    subject: string;
    bodyText: string;
    receivedAt: Date;
  };
  classification: EmailClassification;
  draftResponse?: {
    subject: string;
    bodyText: string;
  };
}

export interface EscalationResult {
  success: boolean;
  messageId?: string;
  error?: string;
}

/**
 * Send an immediate escalation notification
 */
export async function sendEscalationNotification(
  context: EscalationContext
): Promise<EscalationResult> {
  const apiKey = process.env.RESEND_API_KEY;
  
  if (!apiKey) {
    console.error('RESEND_API_KEY not configured');
    return { success: false, error: 'Email service not configured' };
  }

  const urgencyEmoji = getUrgencyEmoji(context.classification.urgency);
  const categoryLabel = getCategoryLabel(context.classification.category);
  
  const subject = `${urgencyEmoji} [${categoryLabel}] ${context.originalEmail.subject}`;
  
  const html = generateEscalationEmailHtml(context);
  const text = generateEscalationEmailText(context);

  try {
    const response = await fetch(RESEND_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: 'Kelly Escalations <hello@curiouskelly.com>',
        to: ESCALATION_EMAIL,
        reply_to: context.originalEmail.from,
        subject,
        html,
        text,
        tags: [
          { name: 'type', value: 'escalation' },
          { name: 'category', value: context.classification.category },
          { name: 'urgency', value: context.classification.urgency },
        ],
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      console.error('Escalation email failed:', data);
      return { success: false, error: data.message || 'Failed to send' };
    }

    console.log(`Escalation sent for thread ${context.threadId}:`, data.id);
    return { success: true, messageId: data.id };

  } catch (error) {
    console.error('Escalation error:', error);
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    };
  }
}

/**
 * Send a daily digest of items needing attention
 */
export async function sendDailyDigest(
  pendingItems: Array<{
    threadId: string;
    from: string;
    subject: string;
    category: string;
    urgency: string;
    receivedAt: Date;
    aiSummary: string;
  }>
): Promise<EscalationResult> {
  const apiKey = process.env.RESEND_API_KEY;
  
  if (!apiKey) {
    return { success: false, error: 'Email service not configured' };
  }

  if (pendingItems.length === 0) {
    console.log('No pending items for daily digest');
    return { success: true };
  }

  const html = generateDigestHtml(pendingItems);
  const text = generateDigestText(pendingItems);

  try {
    const response = await fetch(RESEND_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: 'Kelly <hello@curiouskelly.com>',
        to: ESCALATION_EMAIL,
        subject: `📬 ${pendingItems.length} email${pendingItems.length > 1 ? 's' : ''} need${pendingItems.length === 1 ? 's' : ''} your attention`,
        html,
        text,
        tags: [{ name: 'type', value: 'daily_digest' }],
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      return { success: false, error: data.message || 'Failed to send digest' };
    }

    return { success: true, messageId: data.id };

  } catch (error) {
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    };
  }
}

// ============================================
// EMAIL GENERATION HELPERS
// ============================================

function getUrgencyEmoji(urgency: string): string {
  switch (urgency) {
    case 'critical': return '🚨';
    case 'high': return '⚠️';
    case 'normal': return '📧';
    default: return '📭';
  }
}

function getCategoryLabel(category: string): string {
  const labels: Record<string, string> = {
    support: 'Support',
    billing: 'Billing',
    enterprise: 'Enterprise',
    press: 'Press',
    family: 'Family',
    partner: 'Partner',
    feedback: 'Feedback',
    spam: 'Spam',
    other: 'Other',
  };
  return labels[category] || category;
}

function generateEscalationEmailHtml(context: EscalationContext): string {
  const { originalEmail, classification, draftResponse, threadId } = context;
  
  const adminUrl = `https://curiouskelly.com/admin/email-inbox.html?thread=${threadId}`;
  
  const escalationReasons = classification.escalationTriggers.length > 0
    ? `<ul style="margin: 0; padding-left: 20px; color: #dc2626;">${
        classification.escalationTriggers.map(r => `<li>${escapeHtml(r)}</li>`).join('')
      }</ul>`
    : '<em>Standard escalation for this category</em>';

  const draftSection = draftResponse
    ? `
      <div style="margin-top: 24px; padding: 20px; background: #f0fdf4; border-radius: 8px; border: 1px solid #bbf7d0;">
        <h3 style="margin: 0 0 12px 0; color: #166534; font-size: 14px;">📝 Kelly's Draft Response</h3>
        <p style="margin: 0 0 8px 0; font-weight: 600;">${escapeHtml(draftResponse.subject)}</p>
        <div style="white-space: pre-wrap; color: #374151; font-size: 14px; line-height: 1.6;">
${escapeHtml(draftResponse.bodyText)}
        </div>
      </div>
    `
    : '';

  return `
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #1f2937; max-width: 600px; margin: 0 auto; padding: 20px;">
  
  <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); padding: 20px 24px; border-radius: 12px 12px 0 0;">
    <h1 style="color: white; margin: 0; font-size: 18px;">
      ${getUrgencyEmoji(classification.urgency)} Escalation: ${getCategoryLabel(classification.category)}
    </h1>
    <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-size: 14px;">
      This email needs your attention
    </p>
  </div>
  
  <div style="background: #fef2f2; padding: 24px; border: 1px solid #fecaca; border-top: none; border-radius: 0 0 12px 12px;">
    
    <div style="background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; border: 1px solid #fee2e2;">
      <table style="width: 100%; border-collapse: collapse;">
        <tr>
          <td style="padding: 6px 0; color: #6b7280; width: 100px;">From:</td>
          <td style="padding: 6px 0; font-weight: 600;">${escapeHtml(originalEmail.fromName || originalEmail.from)}</td>
        </tr>
        <tr>
          <td style="padding: 6px 0; color: #6b7280;">Email:</td>
          <td style="padding: 6px 0;"><a href="mailto:${escapeHtml(originalEmail.from)}" style="color: #3b82f6;">${escapeHtml(originalEmail.from)}</a></td>
        </tr>
        <tr>
          <td style="padding: 6px 0; color: #6b7280;">Subject:</td>
          <td style="padding: 6px 0; font-weight: 500;">${escapeHtml(originalEmail.subject)}</td>
        </tr>
        <tr>
          <td style="padding: 6px 0; color: #6b7280;">Received:</td>
          <td style="padding: 6px 0;">${originalEmail.receivedAt.toLocaleString()}</td>
        </tr>
      </table>
    </div>

    <div style="margin-bottom: 20px;">
      <h3 style="margin: 0 0 8px 0; font-size: 14px; color: #374151;">🤖 Kelly's Analysis</h3>
      <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px;">
        <span style="background: #dbeafe; color: #1d4ed8; padding: 4px 10px; border-radius: 12px; font-size: 12px;">
          ${getCategoryLabel(classification.category)}
        </span>
        <span style="background: #fef3c7; color: #92400e; padding: 4px 10px; border-radius: 12px; font-size: 12px;">
          ${classification.sentiment}
        </span>
        <span style="background: #fee2e2; color: #dc2626; padding: 4px 10px; border-radius: 12px; font-size: 12px;">
          ${classification.urgency} urgency
        </span>
      </div>
      <p style="margin: 0; color: #4b5563; font-size: 14px;">${escapeHtml(classification.summary)}</p>
    </div>

    <div style="margin-bottom: 20px;">
      <h3 style="margin: 0 0 8px 0; font-size: 14px; color: #dc2626;">⚠️ Escalation Reasons</h3>
      ${escalationReasons}
    </div>

    <div style="padding: 16px; background: white; border-radius: 8px; border: 1px solid #e5e7eb;">
      <h3 style="margin: 0 0 12px 0; font-size: 14px; color: #374151;">Original Message</h3>
      <div style="white-space: pre-wrap; color: #4b5563; font-size: 14px; line-height: 1.6; max-height: 300px; overflow-y: auto;">
${escapeHtml(originalEmail.bodyText.slice(0, 2000))}${originalEmail.bodyText.length > 2000 ? '\n\n[Message truncated...]' : ''}
      </div>
    </div>

    ${draftSection}

    <div style="margin-top: 24px; display: flex; gap: 12px; flex-wrap: wrap;">
      <a href="${adminUrl}" 
         style="display: inline-block; background: #3b82f6; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600;">
        Review in Admin →
      </a>
      <a href="mailto:${escapeHtml(originalEmail.from)}?subject=Re: ${encodeURIComponent(originalEmail.subject)}" 
         style="display: inline-block; background: #f3f4f6; color: #374151; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; border: 1px solid #d1d5db;">
        Reply Directly
      </a>
    </div>

  </div>
  
  <p style="text-align: center; color: #9ca3af; font-size: 12px; margin-top: 24px;">
    Kelly's Escalation System • ${new Date().toLocaleDateString()}
  </p>
</body>
</html>`.trim();
}

function generateEscalationEmailText(context: EscalationContext): string {
  const { originalEmail, classification, threadId } = context;
  
  return `
ESCALATION: ${getCategoryLabel(classification.category).toUpperCase()}
${'='.repeat(50)}

From: ${originalEmail.fromName || originalEmail.from}
Email: ${originalEmail.from}
Subject: ${originalEmail.subject}
Received: ${originalEmail.receivedAt.toLocaleString()}

KELLY'S ANALYSIS
----------------
Category: ${classification.category}
Sentiment: ${classification.sentiment}
Urgency: ${classification.urgency}
Summary: ${classification.summary}

ESCALATION REASONS
------------------
${classification.escalationTriggers.join('\n') || 'Standard escalation for this category'}

ORIGINAL MESSAGE
----------------
${originalEmail.bodyText.slice(0, 2000)}${originalEmail.bodyText.length > 2000 ? '\n\n[Message truncated...]' : ''}

---
Review in Admin: https://curiouskelly.com/admin/email-inbox.html?thread=${threadId}
Reply: mailto:${originalEmail.from}

Kelly's Escalation System
`.trim();
}

function generateDigestHtml(items: Array<{
  threadId: string;
  from: string;
  subject: string;
  category: string;
  urgency: string;
  receivedAt: Date;
  aiSummary: string;
}>): string {
  const rows = items.map(item => `
    <tr>
      <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
        <span style="background: ${getUrgencyColor(item.urgency)}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">
          ${item.urgency.toUpperCase()}
        </span>
      </td>
      <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
        <span style="background: #f3f4f6; color: #374151; padding: 2px 8px; border-radius: 4px; font-size: 11px;">
          ${getCategoryLabel(item.category)}
        </span>
      </td>
      <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
        <div style="font-weight: 500; color: #1f2937;">${escapeHtml(item.subject)}</div>
        <div style="font-size: 13px; color: #6b7280;">${escapeHtml(item.from)}</div>
      </td>
      <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; color: #6b7280; font-size: 13px;">
        ${item.receivedAt.toLocaleString()}
      </td>
      <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
        <a href="https://curiouskelly.com/admin/email-inbox.html?thread=${item.threadId}" 
           style="color: #3b82f6; font-weight: 500; font-size: 13px;">
          Review →
        </a>
      </td>
    </tr>
  `).join('');

  return `
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #1f2937; max-width: 800px; margin: 0 auto; padding: 20px;">
  
  <div style="text-align: center; margin-bottom: 32px;">
    <h1 style="margin: 0; font-size: 24px;">📬 Daily Email Digest</h1>
    <p style="color: #6b7280; margin: 8px 0 0 0;">
      ${items.length} email${items.length > 1 ? 's' : ''} waiting for your review
    </p>
  </div>

  <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 16px rgba(0,0,0,0.08);">
    <thead>
      <tr style="background: #f9fafb;">
        <th style="padding: 12px; text-align: left; font-size: 12px; color: #6b7280; border-bottom: 1px solid #e5e7eb;">Urgency</th>
        <th style="padding: 12px; text-align: left; font-size: 12px; color: #6b7280; border-bottom: 1px solid #e5e7eb;">Category</th>
        <th style="padding: 12px; text-align: left; font-size: 12px; color: #6b7280; border-bottom: 1px solid #e5e7eb;">Email</th>
        <th style="padding: 12px; text-align: left; font-size: 12px; color: #6b7280; border-bottom: 1px solid #e5e7eb;">Received</th>
        <th style="padding: 12px; text-align: left; font-size: 12px; color: #6b7280; border-bottom: 1px solid #e5e7eb;">Action</th>
      </tr>
    </thead>
    <tbody>
      ${rows}
    </tbody>
  </table>

  <div style="text-align: center; margin-top: 32px;">
    <a href="https://curiouskelly.com/admin/email-inbox.html" 
       style="display: inline-block; background: #3b82f6; color: white; padding: 14px 28px; border-radius: 8px; text-decoration: none; font-weight: 600;">
      Open Email Dashboard →
    </a>
  </div>

  <p style="text-align: center; color: #9ca3af; font-size: 12px; margin-top: 32px;">
    ✨ Kelly's Email System • ${new Date().toLocaleDateString()}
  </p>
</body>
</html>`.trim();
}

function generateDigestText(items: Array<{
  threadId: string;
  from: string;
  subject: string;
  category: string;
  urgency: string;
  receivedAt: Date;
  aiSummary: string;
}>): string {
  const itemList = items.map((item, i) => `
${i + 1}. [${item.urgency.toUpperCase()}] ${item.subject}
   From: ${item.from}
   Category: ${item.category}
   Received: ${item.receivedAt.toLocaleString()}
   Review: https://curiouskelly.com/admin/email-inbox.html?thread=${item.threadId}
`).join('\n');

  return `
DAILY EMAIL DIGEST
==================

${items.length} email(s) need your attention:

${itemList}

---
Open Dashboard: https://curiouskelly.com/admin/email-inbox.html

Kelly's Email System
`.trim();
}

function getUrgencyColor(urgency: string): string {
  switch (urgency) {
    case 'critical': return '#dc2626';
    case 'high': return '#f59e0b';
    case 'normal': return '#3b82f6';
    default: return '#6b7280';
  }
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
