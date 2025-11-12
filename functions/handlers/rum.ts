interface HandlerContext {
  env: Record<string, string | undefined>;
  requestId?: string;
}

interface RumPayload {
  metric: 'LCP' | 'CLS' | 'INP';
  value: number;
  navigationType: string;
  locale: string;
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

async function logRum(payload: unknown) {
  if (typeof process === 'undefined' || process.env.NODE_ENV === 'production') {
    return;
  }
  try {
    const fs = await import('node:fs/promises');
    const dir = '.data';
    await fs.mkdir(dir, { recursive: true });
    await fs.appendFile(`${dir}/rum.log`, `${JSON.stringify(payload)}\n`, 'utf-8');
  } catch (error) {
    console.warn('[rumHandler] Failed to log payload', error);
  }
}

export async function rumHandler(request: Request, context: HandlerContext): Promise<Response> {
  const requestId = context.requestId ?? crypto.randomUUID();
  if (context.env.PUBLIC_RUM_ENABLED !== 'true') {
    return jsonResponse(
      { status: 'disabled', requestId },
      { status: 204, headers: { 'Cache-Control': 'no-store' } }
    );
  }

  if (request.method !== 'POST') {
    return jsonResponse(
      { status: 'error', message: 'method_not_allowed', requestId },
      { status: 405 }
    );
  }

  let body: RumPayload;
  try {
    body = (await request.json()) as RumPayload;
  } catch {
    return jsonResponse(
      { status: 'error', message: 'invalid_json', requestId },
      { status: 400 }
    );
  }

  if (!['LCP', 'CLS', 'INP'].includes(body.metric) || Number.isNaN(Number(body.value))) {
    return jsonResponse(
      { status: 'error', message: 'invalid_payload', requestId },
      { status: 422 }
    );
  }

  await logRum({
    ...body,
    receivedAt: new Date().toISOString(),
    requestId,
    ua: request.headers.get('user-agent')
  });

  return jsonResponse({ status: 'ok', requestId });
}




