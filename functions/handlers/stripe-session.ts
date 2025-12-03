interface HandlerContext {
  env: Record<string, string | undefined>;
  requestId?: string;
}

interface SessionResponse {
  id: string;
  payment_status: string;
  customer_email?: string;
  amount_total?: number;
  currency?: string;
  metadata?: Record<string, string>;
}

type SessionErrorResponse = {
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

export async function stripeSessionHandler(
  request: Request,
  context: HandlerContext
): Promise<Response> {
  const requestId = context.requestId ?? crypto.randomUUID();

  if (request.method !== 'GET') {
    return jsonResponse<SessionErrorResponse>(
      { status: 'error', message: 'method_not_allowed', requestId },
      { status: 405 }
    );
  }

  const stripeKey = context.env.STRIPE_SECRET_KEY;
  if (!stripeKey) {
    return jsonResponse<SessionErrorResponse>(
      { status: 'error', message: 'stripe_not_configured', requestId },
      { status: 503 }
    );
  }

  // Dynamic import to avoid loading Stripe in environments where it's not available
  let Stripe: typeof import('stripe').default;
  try {
    const stripeModule = await import('stripe');
    Stripe = stripeModule.default;
  } catch (error) {
    return jsonResponse<SessionErrorResponse>(
      {
        status: 'error',
        message: 'stripe_module_unavailable',
        requestId,
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 503 }
    );
  }

  const stripe = new Stripe(stripeKey, {
    apiVersion: '2024-11-20.acacia',
  });

  // Get session_id from URL query params
  const url = new URL(request.url);
  const sessionId = url.searchParams.get('session_id');

  if (!sessionId) {
    return jsonResponse<SessionErrorResponse>(
      { status: 'error', message: 'missing_session_id', requestId },
      { status: 400 }
    );
  }

  try {
    const session = await stripe.checkout.sessions.retrieve(sessionId, {
      expand: ['customer', 'subscription'],
    });

    console.info('[stripeSessionHandler] session retrieved', {
      requestId,
      sessionId: session.id,
      paymentStatus: session.payment_status,
    });

    return jsonResponse<SessionResponse>({
      id: session.id,
      payment_status: session.payment_status,
      customer_email: session.customer_email || (session.customer_details?.email as string | undefined),
      amount_total: session.amount_total || undefined,
      currency: session.currency || undefined,
      metadata: session.metadata || undefined,
    });
  } catch (error) {
    console.error('[stripeSessionHandler] error', { requestId, error });
    return jsonResponse<SessionErrorResponse>(
      {
        status: 'error',
        message: 'session_retrieval_failed',
        requestId,
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}












