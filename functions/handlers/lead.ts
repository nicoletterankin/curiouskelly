interface HandlerContext {
  env: Record<string, string | undefined>;
  requestId?: string;
}

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

interface LeadResponse {
  status: 'ok';
  requestId: string;
}

type LeadErrorResponse = {
  status: 'error';
  requestId: string;
  errors?: Record<string, string>;
  message: string;
};

interface TurnstileVerification {
  success: boolean;
  'error-codes'?: string[];
}

interface RecaptchaVerification {
  success: boolean;
  score?: number;
  'error-codes'?: string[];
}

function jsonResponse<T>(body: T, init?: ResponseInit) {
  return new Response(JSON.stringify(body), {
    status: init?.status ?? 200,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'no-store',
      ...(init?.headers ?? {})
    }
  });
}

async function hashValue(value: string): Promise<string> {
  const encoder = new TextEncoder();
  if (typeof crypto !== 'undefined' && crypto.subtle) {
    const hashBuffer = await crypto.subtle.digest('SHA-256', encoder.encode(value));
    return Array.from(new Uint8Array(hashBuffer))
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('')
      .slice(0, 12);
  }
  const nodeCrypto = await import('node:crypto');
  return nodeCrypto.createHash('sha256').update(value).digest('hex').slice(0, 12);
}

async function logToFile(fileName: string, payload: unknown) {
  if (typeof process === 'undefined' || process.env.NODE_ENV === 'production') {
    return;
  }
  try {
    const fs = await import('node:fs/promises');
    const dir = '.data';
    await fs.mkdir(dir, { recursive: true });
    await fs.appendFile(`${dir}/${fileName}`, `${JSON.stringify(payload)}\n`, 'utf-8');
  } catch (error) {
    console.warn('[leadHandler] Failed to log payload', error);
  }
}

function isValidName(value: unknown) {
  return typeof value === 'string' && /^[\p{L}\p{M}' -]{2,}$/u.test(value.trim());
}

function isValidEmail(value: unknown) {
  return typeof value === 'string' && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value.trim().toLowerCase());
}

function isValidPhone(value: unknown) {
  return typeof value === 'string' && value.trim().length >= 6;
}

function collectValidationErrors(body: LeadPayload) {
  const errors: Record<string, string> = {};
  if (!isValidName(body.first_name)) {
    errors.first_name = 'first_name_invalid';
  }
  if (!isValidName(body.last_name)) {
    errors.last_name = 'last_name_invalid';
  }
  if (!isValidEmail(body.email)) {
    errors.email = 'email_invalid';
  }
  if (!isValidPhone(body.phone)) {
    errors.phone = 'phone_invalid';
  }
  if (!body.country) {
    errors.country = 'country_required';
  }
  if (!body.region) {
    errors.region = 'region_required';
  }
  if (!body.locale) {
    errors.locale = 'locale_required';
  }
  if (!body.journey) {
    errors.journey = 'journey_required';
  }
  return errors;
}

async function verifyTurnstile(token: string | undefined, secret: string) {
  if (!token) {
    return { success: false, message: 'turnstile_missing_token' };
  }
  const params = new URLSearchParams();
  params.append('secret', secret);
  params.append('response', token);
  const response = await fetch('https://challenges.cloudflare.com/turnstile/v0/siteverify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: params
  });
  const result = (await response.json()) as TurnstileVerification;
  if (!result.success) {
    return { success: false, message: result['error-codes']?.join(',') ?? 'turnstile_failed' };
  }
  return { success: true };
}

async function verifyRecaptcha(token: string | undefined, secret: string) {
  if (!token) {
    return { success: false, message: 'recaptcha_missing_token' };
  }
  const params = new URLSearchParams();
  params.append('secret', secret);
  params.append('response', token);
  const response = await fetch('https://www.google.com/recaptcha/api/siteverify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: params
  });
  const result = (await response.json()) as RecaptchaVerification;
  if (!result.success) {
    return { success: false, message: result['error-codes']?.join(',') ?? 'recaptcha_failed' };
  }
  if (result.score !== undefined && result.score < 0.5) {
    return { success: false, message: 'recaptcha_low_score' };
  }
  return { success: true };
}

async function postToCrm(url: string, authToken: string | undefined, payload: unknown, timeoutMs: number) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(authToken ? { Authorization: `Bearer ${authToken}` } : {})
      },
      body: JSON.stringify(payload),
      signal: controller.signal
    });
    if (!response.ok) {
      throw new Error(`CRM responded with ${response.status}`);
    }
  } finally {
    clearTimeout(timeout);
  }
}

export async function leadHandler(request: Request, context: HandlerContext): Promise<Response> {
  const start = typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
  const requestId = context.requestId ?? crypto.randomUUID();

  if (request.method !== 'POST') {
    return jsonResponse<LeadErrorResponse>(
      { status: 'error', message: 'method_not_allowed', requestId },
      { status: 405 }
    );
  }

  let body: LeadPayload & { turnstile_token?: string; recaptcha_token?: string };
  try {
    body = (await request.json()) as LeadPayload & {
      turnstile_token?: string;
      recaptcha_token?: string;
    };
  } catch {
    return jsonResponse<LeadErrorResponse>(
      { status: 'error', message: 'invalid_json', requestId },
      { status: 400 }
    );
  }

  const validationErrors = collectValidationErrors(body);
  if (Object.keys(validationErrors).length > 0) {
    return jsonResponse<LeadErrorResponse>(
      { status: 'error', message: 'validation_failed', requestId, errors: validationErrors },
      { status: 422 }
    );
  }

  const turnstileSecret = context.env.TURNSTILE_SECRET_KEY;
  const recaptchaSecret = context.env.RECAPTCHA_SECRET_KEY;

  if (turnstileSecret) {
    const verification = await verifyTurnstile(body.turnstile_token, turnstileSecret);
    if (!verification.success) {
      return jsonResponse<LeadErrorResponse>(
        { status: 'error', message: verification.message, requestId },
        { status: 400 }
      );
    }
  } else if (recaptchaSecret) {
    const verification = await verifyRecaptcha(body.recaptcha_token, recaptchaSecret);
    if (!verification.success) {
      return jsonResponse<LeadErrorResponse>(
        { status: 'error', message: verification.message, requestId },
        { status: 400 }
      );
    }
  }

  const crmWebhookUrl = context.env.CRM_WEBHOOK_URL;
  const crmAuthToken = context.env.CRM_AUTH_TOKEN;
  const crmTimeout = Number.parseInt(context.env.CRM_TIMEOUT_MS ?? '5000', 10);
  const sanitizedPayload = {
    first_name: body.first_name,
    last_name: body.last_name,
    email: body.email,
    phone: body.phone,
    country: body.country,
    region: body.region,
    marketing_opt_in: Boolean(body.marketing_opt_in),
    locale: body.locale,
    journey: body.journey,
    submitted_at: new Date().toISOString(),
    request_id: requestId,
    user_agent: request.headers.get('user-agent') ?? undefined,
    referer: request.headers.get('referer') ?? undefined
  };

  const emailHash = await hashValue(body.email.toLowerCase());
  console.info('[leadHandler] lead received', {
    requestId,
    emailHash,
    locale: body.locale,
    journey: body.journey
  });

  if (crmWebhookUrl) {
    try {
      await postToCrm(crmWebhookUrl, crmAuthToken, sanitizedPayload, crmTimeout);
    } catch (error) {
      console.error('[leadHandler] CRM forward failed', { requestId, error });
      return jsonResponse<LeadErrorResponse>(
        { status: 'error', message: 'crm_unavailable', requestId },
        { status: 502 }
      );
    }
  }

  await logToFile('leads.log', sanitizedPayload);

  const end = typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
  console.info('[leadHandler] completed', {
    requestId,
    durationMs: Math.round(end - start)
  });

  return jsonResponse<LeadResponse>({ status: 'ok', requestId });
}

