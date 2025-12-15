import type { VercelRequest, VercelResponse } from '@vercel/node';

/**
 * Institutional / B2B Lead Intake
 * POST /api/institutional-lead
 *
 * Creates a sales-assisted “checkout” by sending a lead to hello@curiouskelly.com.
 * (No database schema changes required.)
 */

const RESEND_API_URL = 'https://api.resend.com/emails';

type OrganizationType = 'classroom' | 'school' | 'district' | 'enterprise';

interface InstitutionalLeadRequest {
  organizationType: OrganizationType;
  organizationName: string;
  seats: number;
  contactName: string;
  contactEmail: string;
  billingEmail?: string;
  purchaseOrderNumber?: string;
  taxExempt?: boolean;
  taxExemptId?: string;
  notes?: string;
}

function isValidEmail(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim().toLowerCase());
}

function escapeHtml(text: string): string {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ ok: false, error: 'method_not_allowed' });

  const resendKey = process.env.RESEND_API_KEY;
  if (!resendKey) return res.status(503).json({ ok: false, error: 'resend_not_configured' });

  const body = (req.body || {}) as InstitutionalLeadRequest;

  if (!body.organizationType || !['classroom', 'school', 'district', 'enterprise'].includes(body.organizationType)) {
    return res.status(422).json({ ok: false, error: 'invalid_organization_type' });
  }
  if (!body.organizationName || String(body.organizationName).trim().length < 2) {
    return res.status(422).json({ ok: false, error: 'organization_name_required' });
  }
  if (!Number.isFinite(body.seats) || body.seats <= 0) {
    return res.status(422).json({ ok: false, error: 'invalid_seats' });
  }
  if (!body.contactName || String(body.contactName).trim().length < 2) {
    return res.status(422).json({ ok: false, error: 'contact_name_required' });
  }
  if (!body.contactEmail || !isValidEmail(body.contactEmail)) {
    return res.status(422).json({ ok: false, error: 'invalid_contact_email' });
  }
  if (body.billingEmail && !isValidEmail(body.billingEmail)) {
    return res.status(422).json({ ok: false, error: 'invalid_billing_email' });
  }

  const payloadHtml = `
    <div style="font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; line-height: 1.5; color: #111;">
      <h2 style="margin: 0 0 12px 0;">Institutional Lead</h2>
      <table style="border-collapse: collapse; width: 100%; max-width: 760px;">
        <tr><td style="padding: 6px 10px; border-bottom: 1px solid #eee;"><b>Org type</b></td><td style="padding: 6px 10px; border-bottom: 1px solid #eee;">${escapeHtml(body.organizationType)}</td></tr>
        <tr><td style="padding: 6px 10px; border-bottom: 1px solid #eee;"><b>Org name</b></td><td style="padding: 6px 10px; border-bottom: 1px solid #eee;">${escapeHtml(body.organizationName)}</td></tr>
        <tr><td style="padding: 6px 10px; border-bottom: 1px solid #eee;"><b>Seats</b></td><td style="padding: 6px 10px; border-bottom: 1px solid #eee;">${escapeHtml(String(body.seats))}</td></tr>
        <tr><td style="padding: 6px 10px; border-bottom: 1px solid #eee;"><b>Contact</b></td><td style="padding: 6px 10px; border-bottom: 1px solid #eee;">${escapeHtml(body.contactName)} — ${escapeHtml(body.contactEmail)}</td></tr>
        <tr><td style="padding: 6px 10px; border-bottom: 1px solid #eee;"><b>Billing email</b></td><td style="padding: 6px 10px; border-bottom: 1px solid #eee;">${escapeHtml(body.billingEmail || '')}</td></tr>
        <tr><td style="padding: 6px 10px; border-bottom: 1px solid #eee;"><b>PO #</b></td><td style="padding: 6px 10px; border-bottom: 1px solid #eee;">${escapeHtml(body.purchaseOrderNumber || '')}</td></tr>
        <tr><td style="padding: 6px 10px; border-bottom: 1px solid #eee;"><b>Tax exempt</b></td><td style="padding: 6px 10px; border-bottom: 1px solid #eee;">${escapeHtml(String(!!body.taxExempt))}</td></tr>
        <tr><td style="padding: 6px 10px; border-bottom: 1px solid #eee;"><b>Tax ID</b></td><td style="padding: 6px 10px; border-bottom: 1px solid #eee;">${escapeHtml(body.taxExemptId || '')}</td></tr>
        <tr><td style="padding: 6px 10px; border-bottom: 1px solid #eee;"><b>Notes</b></td><td style="padding: 6px 10px; border-bottom: 1px solid #eee;">${escapeHtml(body.notes || '')}</td></tr>
      </table>
      <p style="margin-top: 14px; color: #555;">Submitted from: ${escapeHtml(req.headers['referer'] as string || 'unknown')}</p>
    </div>
  `;

  try {
    const emailRes = await fetch(RESEND_API_URL, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${resendKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: 'Curious Kelly <hello@curiouskelly.com>',
        to: ['hello@curiouskelly.com'],
        subject: `[Institutional] ${body.organizationType.toUpperCase()} — ${body.organizationName} (${body.seats} seats)`,
        reply_to: body.contactEmail,
        html: payloadHtml,
      }),
    });

    if (!emailRes.ok) {
      const details = await emailRes.text();
      return res.status(502).json({ ok: false, error: 'email_failed', details });
    }

    return res.status(200).json({ ok: true });
  } catch (e) {
    console.error('institutional-lead error:', e);
    return res.status(500).json({ ok: false, error: 'internal_error' });
  }
}

