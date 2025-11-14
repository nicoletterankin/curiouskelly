import type { APIRoute } from 'astro';
import { parsePhoneNumberFromString } from 'libphonenumber-js/min';

// Inlined validation functions to avoid module resolution issues in Vercel builds
interface LeadPayload {
  first_name: string;
  last_name: string;
  email: string;
  phone: string;
  country: string;
  region: string;
  marketing_opt_in: boolean;
  locale: string;
  journey: string;
}

type LeadFormData = LeadPayload;

type LeadErrors = Partial<Record<keyof LeadPayload | 'turnstile', string>>;

interface LeadFormCopy {
  form: {
    errors: {
      required: string;
      email: string;
      phone: string;
      minLength: string;
      maxLength: string;
    };
  };
}

const namePattern = /^[\p{L}\p{M}' -]{2,}$/u;
const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function sanitizeFormData(body: any): LeadPayload {
  return {
    first_name: String(body.first_name || '').trim(),
    last_name: String(body.last_name || '').trim(),
    email: String(body.email || '').trim().toLowerCase(),
    phone: String(body.phone || '').trim(),
    country: String(body.country || '').trim(),
    region: String(body.region || '').trim(),
    marketing_opt_in: Boolean(body.marketing_opt_in),
    locale: String(body.locale || 'en-US').trim(),
    journey: String(body.journey || '').trim()
  };
}

function validateLeadForm(data: LeadPayload, copy: LeadFormCopy): { valid: boolean; errors: LeadErrors } {
  const errors: LeadErrors = {};

  if (!data.first_name || !namePattern.test(data.first_name.trim())) {
    errors.first_name = data.first_name ? copy.form.errors.minLength : copy.form.errors.required;
  }

  if (!data.last_name || !namePattern.test(data.last_name.trim())) {
    errors.last_name = data.last_name ? copy.form.errors.minLength : copy.form.errors.required;
  }

  if (!data.email || !emailPattern.test(data.email.trim().toLowerCase())) {
    errors.email = data.email ? copy.form.errors.email : copy.form.errors.required;
  }

  if (!data.country) {
    errors.country = copy.form.errors.required;
  }

  if (!data.region) {
    errors.region = copy.form.errors.required;
  }

  if (!data.phone) {
    errors.phone = copy.form.errors.required;
  } else {
    const parsed = parsePhoneNumberFromString(data.phone);
    if (!parsed || !parsed.isValid()) {
      errors.phone = copy.form.errors.phone;
    }
  }

  if (!data.locale) {
    errors.locale = copy.form.errors.required;
  }

  if (!data.journey) {
    errors.journey = copy.form.errors.required;
  }

  return {
    valid: Object.keys(errors).length === 0,
    errors
  };
}

export const POST: APIRoute = async ({ request }) => {
  try {
    const body = await request.json();
    const data = sanitizeFormData(body);

    // Validate Turnstile token
    const turnstileToken = body.turnstile_token;
    const turnstileSecret = import.meta.env.TURNSTILE_SECRET_KEY;
    const turnstileSiteKey = import.meta.env.PUBLIC_TURNSTILE_SITE_KEY;

    if (turnstileSecret && turnstileSiteKey && turnstileToken) {
      const verifyResponse = await fetch('https://challenges.cloudflare.com/turnstile/v0/siteverify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          secret: turnstileSecret,
          response: turnstileToken
        })
      });

      const verifyResult = await verifyResponse.json();
      if (!verifyResult.success) {
        return new Response(
          JSON.stringify({ success: false, error: 'Verification failed' }),
          { status: 400, headers: { 'Content-Type': 'application/json' } }
        );
      }
    }

    // Validate form data (server-side)
    const t = {
      form: {
        errors: {
          required: 'This field is required',
          email: 'Invalid email',
          phone: 'Invalid phone',
          minLength: 'Too short',
          maxLength: 'Too long'
        }
      }
    };

    const validation = validateLeadForm(data, t);
    if (!validation.valid) {
      return new Response(
        JSON.stringify({ success: false, errors: validation.errors }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Forward to CRM webhook if configured
    const crmWebhookUrl = import.meta.env.CRM_WEBHOOK_URL;
    if (crmWebhookUrl) {
      try {
        const crmResponse = await fetch(crmWebhookUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ...data,
            source: 'website',
            timestamp: new Date().toISOString()
          })
        });

        if (!crmResponse.ok) {
          console.error('CRM webhook failed:', await crmResponse.text());
        }
      } catch (error: any) {
        console.error('CRM webhook error:', error.message);
        // Don't fail the request if CRM fails
      }
    }

    // Log to file store in dev (placeholder)
    if (import.meta.env.DEV) {
      console.log('Lead submitted:', data);
    }

    return new Response(
      JSON.stringify({ success: true, message: 'Lead submitted successfully' }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    );
  } catch (error: any) {
    console.error('Lead submission error:', error);
    return new Response(
      JSON.stringify({ success: false, error: error.message || 'Internal server error' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
};











