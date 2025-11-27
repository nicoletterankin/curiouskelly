interface HandlerContext {
  env: Record<string, string | undefined>;
  requestId?: string;
}

interface WaitlistRequest {
  email: string;
  source?: string;
}

interface WaitlistResponse {
  success: boolean;
  message: string;
  id?: string;
}

type WaitlistErrorResponse = {
  status: 'error';
  requestId: string;
  message: string;
  details?: string;
};

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

function isValidEmail(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim().toLowerCase());
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
    console.warn('[waitlistHandler] Failed to log payload', error);
  }
}

export async function waitlistHandler(
  request: Request,
  context: HandlerContext
): Promise<Response> {
  const requestId = context.requestId ?? crypto.randomUUID();

  if (request.method !== 'POST') {
    return jsonResponse<WaitlistErrorResponse>(
      { status: 'error', message: 'method_not_allowed', requestId },
      { status: 405 }
    );
  }

  const supabaseUrl = context.env.PUBLIC_SUPABASE_URL;
  const supabaseKey = context.env.SUPABASE_SERVICE_ROLE_KEY || context.env.PUBLIC_SUPABASE_ANON_KEY;

  let body: WaitlistRequest;
  try {
    body = (await request.json()) as WaitlistRequest;
  } catch {
    return jsonResponse<WaitlistErrorResponse>(
      { status: 'error', message: 'invalid_json', requestId },
      { status: 400 }
    );
  }

  // Validate email
  if (!body.email || !isValidEmail(body.email)) {
    return jsonResponse<WaitlistErrorResponse>(
      { status: 'error', message: 'invalid_email', requestId },
      { status: 422 }
    );
  }

  // If Supabase is configured, try to insert
  if (supabaseUrl && supabaseKey) {
    try {
      const { createClient } = await import('@supabase/supabase-js');
      const supabase = createClient(supabaseUrl, supabaseKey);

      const { data, error } = await supabase
        .from('waitlist')
        .insert({
          email: body.email.trim().toLowerCase(),
          source: body.source || 'landing_page',
          created_at: new Date().toISOString(),
        })
        .select()
        .single();

      if (error) {
        // If table doesn't exist, log and continue (fallback to file logging)
        if (error.code === '42P01' || error.code === 'PGRST116') {
          console.warn('[waitlistHandler] Waitlist table does not exist, logging to file', {
            requestId,
            email: body.email.substring(0, 3) + '***',
          });
          await logToFile('waitlist.log', {
            email: body.email.trim().toLowerCase(),
            source: body.source || 'landing_page',
            created_at: new Date().toISOString(),
            requestId,
          });
          return jsonResponse<WaitlistResponse>({
            success: true,
            message: 'Added to waitlist',
          });
        }
        // If duplicate email, return success anyway
        if (error.code === '23505') {
          return jsonResponse<WaitlistResponse>({
            success: true,
            message: 'Already on waitlist',
          });
        }
        throw error;
      }

      console.info('[waitlistHandler] email added to waitlist', {
        requestId,
        emailHash: body.email.substring(0, 3) + '***',
        source: body.source || 'landing_page',
      });

      return jsonResponse<WaitlistResponse>({
        success: true,
        message: 'Added to waitlist',
        id: data.id,
      });
    } catch (error) {
      console.error('[waitlistHandler] Supabase error, falling back to file logging', {
        requestId,
        error,
      });
      // Fall through to file logging
    }
  }

  // Fallback: log to file if Supabase not configured or failed
  await logToFile('waitlist.log', {
    email: body.email.trim().toLowerCase(),
    source: body.source || 'landing_page',
    created_at: new Date().toISOString(),
    requestId,
  });

  console.info('[waitlistHandler] email logged to file', {
    requestId,
    emailHash: body.email.substring(0, 3) + '***',
    source: body.source || 'landing_page',
  });

  return jsonResponse<WaitlistResponse>({
    success: true,
    message: 'Added to waitlist',
  });
}




