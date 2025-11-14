import type { APIRoute } from 'astro';
import { validateLeadForm, sanitizeFormData, type LeadFormData } from '@lib/validation';

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











