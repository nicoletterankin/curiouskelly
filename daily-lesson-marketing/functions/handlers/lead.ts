// Core handler logic (framework-agnostic)
import { validateLeadForm, sanitizeFormData, type LeadFormData } from '../../lib/validation';

export interface LeadRequest {
  body: any;
  headers: Record<string, string>;
}

export interface LeadResponse {
  status: number;
  body: any;
}

export async function handleLeadRequest(req: LeadRequest): Promise<LeadResponse> {
  try {
    const data = sanitizeFormData(req.body);

    // Validate Turnstile token
    const turnstileToken = req.body.turnstile_token;
    const turnstileSecret = process.env.TURNSTILE_SECRET_KEY;
    const turnstileSiteKey = process.env.PUBLIC_TURNSTILE_SITE_KEY;

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
        return {
          status: 400,
          body: { success: false, error: 'Verification failed' }
        };
      }
    }

    // Validate form data
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
      return {
        status: 400,
        body: { success: false, errors: validation.errors }
      };
    }

    // Forward to CRM webhook if configured
    const crmWebhookUrl = process.env.CRM_WEBHOOK_URL;
    if (crmWebhookUrl) {
      try {
        await fetch(crmWebhookUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ...data,
            source: 'website',
            timestamp: new Date().toISOString()
          })
        });
      } catch (error: any) {
        console.error('CRM webhook error:', error.message);
      }
    }

    return {
      status: 200,
      body: { success: true, message: 'Lead submitted successfully' }
    };
  } catch (error: any) {
    return {
      status: 500,
      body: { success: false, error: error.message || 'Internal server error' }
    };
  }
}











