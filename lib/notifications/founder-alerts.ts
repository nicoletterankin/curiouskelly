/**
 * FOUNDER ALERTS
 * 
 * Email notifications to hello@curiouskelly.com for:
 * - 🎉 Happy learner events (celebrations)
 * - 🚨 Critical issues requiring intervention
 * - 📊 Weekly digest of platform health
 * 
 * Philosophy: Remove founder from day-to-day. Only escalate what
 * the community can't solve itself.
 */

import { createClient, SupabaseClient } from '@supabase/supabase-js';

const FOUNDER_EMAIL = 'hello@curiouskelly.com';
const FROM_EMAIL = 'hello@curiouskelly.com';
const SENDGRID_API_KEY = process.env.SENDGRID_API_KEY!;

// ═══════════════════════════════════════════════════════════════════
// EMAIL SENDING
// ═══════════════════════════════════════════════════════════════════

interface EmailPayload {
  subject: string;
  html: string;
  text?: string;
}

export async function sendFounderEmail(payload: EmailPayload): Promise<boolean> {
  try {
    const response = await fetch('https://api.sendgrid.com/v3/mail/send', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${SENDGRID_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        personalizations: [{ to: [{ email: FOUNDER_EMAIL }] }],
        from: { email: FROM_EMAIL, name: 'Kelly (Your AI Teacher)' },
        subject: payload.subject,
        content: [
          { type: 'text/html', value: payload.html },
          ...(payload.text ? [{ type: 'text/plain', value: payload.text }] : [])
        ]
      })
    });

    return response.ok;
  } catch (err) {
    console.error('Failed to send founder email:', err);
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════
// EMAIL TEMPLATES
// ═══════════════════════════════════════════════════════════════════

const baseStyles = `
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8f9fa; margin: 0; padding: 20px; }
  .container { max-width: 600px; margin: 0 auto; background: white; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 24px rgba(0,0,0,0.1); }
  .header { padding: 32px; text-align: center; }
  .header.celebration { background: linear-gradient(135deg, #00b894, #00cec9); color: white; }
  .header.alert { background: linear-gradient(135deg, #e94560, #ff6b6b); color: white; }
  .header.digest { background: linear-gradient(135deg, #635bff, #7a73ff); color: white; }
  .header h1 { margin: 0; font-size: 24px; }
  .header .emoji { font-size: 48px; margin-bottom: 16px; }
  .content { padding: 32px; }
  .stat-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin: 24px 0; }
  .stat { background: #f8f9fa; padding: 16px; border-radius: 12px; text-align: center; }
  .stat .number { font-size: 32px; font-weight: bold; color: #1a1a2e; }
  .stat .label { font-size: 12px; color: #666; text-transform: uppercase; }
  .item { padding: 16px; background: #f8f9fa; border-radius: 8px; margin-bottom: 12px; }
  .item .title { font-weight: 600; margin-bottom: 4px; }
  .item .detail { font-size: 14px; color: #666; }
  .footer { padding: 24px 32px; background: #f8f9fa; text-align: center; font-size: 13px; color: #888; }
  .btn { display: inline-block; padding: 12px 24px; background: #1a1a2e; color: white; text-decoration: none; border-radius: 8px; margin-top: 16px; }
`;

export function celebrationEmail(data: {
  headline: string;
  subhead: string;
  stats: { label: string; value: string | number }[];
  highlights: { title: string; detail: string }[];
}): EmailPayload {
  return {
    subject: `🎉 ${data.headline}`,
    html: `<!DOCTYPE html>
<html>
<head><style>${baseStyles}</style></head>
<body>
<div class="container">
  <div class="header celebration">
    <div class="emoji">🎉</div>
    <h1>${data.headline}</h1>
    <p style="margin-top: 8px; opacity: 0.9;">${data.subhead}</p>
  </div>
  <div class="content">
    <div class="stat-grid">
      ${data.stats.map(s => `
        <div class="stat">
          <div class="number">${s.value}</div>
          <div class="label">${s.label}</div>
        </div>
      `).join('')}
    </div>
    
    ${data.highlights.length > 0 ? `
      <h3 style="margin-top: 24px;">✨ Highlights</h3>
      ${data.highlights.map(h => `
        <div class="item">
          <div class="title">${h.title}</div>
          <div class="detail">${h.detail}</div>
        </div>
      `).join('')}
    ` : ''}
    
    <p style="text-align: center; margin-top: 24px; color: #666;">
      This is why you built this. Every lesson changes a life.
    </p>
  </div>
  <div class="footer">
    From Kelly with love 💚 • Lesson of the Day PBC
  </div>
</div>
</body>
</html>`
  };
}

export function alertEmail(data: {
  severity: 'warning' | 'critical';
  headline: string;
  issues: { title: string; detail: string; actionUrl?: string }[];
  context?: string;
}): EmailPayload {
  const emoji = data.severity === 'critical' ? '🚨' : '⚠️';
  
  return {
    subject: `${emoji} ${data.headline}`,
    html: `<!DOCTYPE html>
<html>
<head><style>${baseStyles}</style></head>
<body>
<div class="container">
  <div class="header alert">
    <div class="emoji">${emoji}</div>
    <h1>${data.headline}</h1>
  </div>
  <div class="content">
    ${data.context ? `<p style="margin-bottom: 24px; color: #666;">${data.context}</p>` : ''}
    
    <h3>Issues Requiring Attention</h3>
    ${data.issues.map(i => `
      <div class="item" style="border-left: 4px solid ${data.severity === 'critical' ? '#e94560' : '#ffc107'};">
        <div class="title">${i.title}</div>
        <div class="detail">${i.detail}</div>
        ${i.actionUrl ? `<a href="${i.actionUrl}" class="btn" style="margin-top: 12px; display: inline-block;">Take Action →</a>` : ''}
      </div>
    `).join('')}
    
    <p style="text-align: center; margin-top: 24px; font-size: 14px; color: #888;">
      Only escalated because the community couldn't resolve it.
    </p>
  </div>
  <div class="footer">
    Curious Kelly Alert System • <a href="https://curiouskelly.com/admin">Mission Control</a>
  </div>
</div>
</body>
</html>`
  };
}

export function weeklyDigestEmail(data: {
  weekNumber: number;
  periodStart: string;
  periodEnd: string;
  stats: {
    newLearners: number;
    lessonsCompleted: number;
    commentsPosted: number;
    suggestionsReceived: number;
    issuesResolved: number;
    issuesPending: number;
  };
  topMoments: { title: string; detail: string }[];
  needsAttention: { title: string; detail: string }[];
}): EmailPayload {
  return {
    subject: `📊 Week ${data.weekNumber} Digest — ${data.stats.newLearners} new learners`,
    html: `<!DOCTYPE html>
<html>
<head><style>${baseStyles}</style></head>
<body>
<div class="container">
  <div class="header digest">
    <div class="emoji">📊</div>
    <h1>Week ${data.weekNumber} Digest</h1>
    <p style="margin-top: 8px; opacity: 0.9;">${data.periodStart} — ${data.periodEnd}</p>
  </div>
  <div class="content">
    <div class="stat-grid">
      <div class="stat">
        <div class="number">${data.stats.newLearners}</div>
        <div class="label">New Learners</div>
      </div>
      <div class="stat">
        <div class="number">${data.stats.lessonsCompleted}</div>
        <div class="label">Lessons Completed</div>
      </div>
      <div class="stat">
        <div class="number">${data.stats.commentsPosted}</div>
        <div class="label">Comments</div>
      </div>
      <div class="stat">
        <div class="number">${data.stats.suggestionsReceived}</div>
        <div class="label">Suggestions</div>
      </div>
    </div>
    
    <div style="display: flex; gap: 16px; margin: 24px 0;">
      <div style="flex: 1; padding: 16px; background: #d4edda; border-radius: 8px; text-align: center;">
        <strong style="color: #155724;">✅ ${data.stats.issuesResolved}</strong>
        <div style="font-size: 12px; color: #155724;">Resolved by Community</div>
      </div>
      <div style="flex: 1; padding: 16px; background: ${data.stats.issuesPending > 0 ? '#fff3cd' : '#d4edda'}; border-radius: 8px; text-align: center;">
        <strong style="color: ${data.stats.issuesPending > 0 ? '#856404' : '#155724'};">${data.stats.issuesPending > 0 ? '⏳' : '✅'} ${data.stats.issuesPending}</strong>
        <div style="font-size: 12px; color: ${data.stats.issuesPending > 0 ? '#856404' : '#155724'};">Pending</div>
      </div>
    </div>
    
    ${data.topMoments.length > 0 ? `
      <h3>🌟 Top Moments</h3>
      ${data.topMoments.slice(0, 3).map(m => `
        <div class="item">
          <div class="title">${m.title}</div>
          <div class="detail">${m.detail}</div>
        </div>
      `).join('')}
    ` : ''}
    
    ${data.needsAttention.length > 0 ? `
      <h3 style="margin-top: 24px;">👀 Needs Your Eyes</h3>
      ${data.needsAttention.map(n => `
        <div class="item" style="border-left: 4px solid #ffc107;">
          <div class="title">${n.title}</div>
          <div class="detail">${n.detail}</div>
        </div>
      `).join('')}
    ` : `
      <div style="text-align: center; padding: 24px; background: #d4edda; border-radius: 12px; margin-top: 24px;">
        <div style="font-size: 32px;">🎉</div>
        <div style="color: #155724; font-weight: 600;">Everything Running Smoothly</div>
        <div style="color: #155724; font-size: 14px;">No issues requiring your attention</div>
      </div>
    `}
    
    <a href="https://curiouskelly.com/admin" class="btn" style="display: block; text-align: center;">View Mission Control →</a>
  </div>
  <div class="footer">
    Weekly digest from Kelly • <a href="https://curiouskelly.com/admin/stripe">Revenue</a> • <a href="https://curiouskelly.com/commons">Commons</a>
  </div>
</div>
</body>
</html>`
  };
}

// ═══════════════════════════════════════════════════════════════════
// EVENT TRIGGERS
// ═══════════════════════════════════════════════════════════════════

export async function notifyMilestone(supabase: SupabaseClient, type: string, count: number): Promise<void> {
  const milestones = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000];
  
  if (!milestones.includes(count)) return;
  
  const milestoneNames: { [key: string]: string } = {
    'learners': 'learners',
    'lessons_completed': 'lessons completed',
    'comments': 'comments posted',
    'subscriptions': 'subscribers'
  };
  
  await sendFounderEmail(celebrationEmail({
    headline: `${count} ${milestoneNames[type] || type}!`,
    subhead: `A meaningful milestone. Thank you, past Nicolette.`,
    stats: [{ label: 'Milestone', value: count }],
    highlights: []
  }));
  
  // Log milestone
  await supabase.from('founder_notifications').insert({
    type: 'milestone',
    data: { milestone_type: type, count },
    sent_at: new Date().toISOString()
  });
}

export async function notifyEscalation(supabase: SupabaseClient, issue: {
  type: 'moderation' | 'bug' | 'payment' | 'suggestion';
  title: string;
  detail: string;
  lesson_day?: number;
  phase?: string;
  user_id?: string;
}): Promise<void> {
  await sendFounderEmail(alertEmail({
    severity: issue.type === 'payment' ? 'critical' : 'warning',
    headline: `Escalation: ${issue.type}`,
    issues: [{
      title: issue.title,
      detail: `${issue.detail}${issue.lesson_day ? ` (Day ${issue.lesson_day}${issue.phase ? `, ${issue.phase}` : ''})` : ''}`,
      actionUrl: 'https://curiouskelly.com/admin'
    }],
    context: 'This was escalated because the community couldn\'t resolve it within 48 hours.'
  }));
  
  await supabase.from('founder_notifications').insert({
    type: 'escalation',
    data: issue,
    sent_at: new Date().toISOString()
  });
}

// ═══════════════════════════════════════════════════════════════════
// HAPPY LEARNER EVENTS (batched daily)
// ═══════════════════════════════════════════════════════════════════

export interface HappyEvent {
  type: 'first_lesson' | 'streak_7' | 'streak_30' | 'streak_100' | 'completed_track' | 'helpful_comment' | 'first_comment';
  user_id: string;
  detail?: string;
  created_at: string;
}

export async function logHappyEvent(supabase: SupabaseClient, event: HappyEvent): Promise<void> {
  await supabase.from('happy_learner_events').insert(event);
}

export async function getHappyEventsSince(supabase: SupabaseClient, since: Date): Promise<HappyEvent[]> {
  const { data } = await supabase
    .from('happy_learner_events')
    .select('*')
    .gte('created_at', since.toISOString())
    .order('created_at', { ascending: false });
  
  return data || [];
}
