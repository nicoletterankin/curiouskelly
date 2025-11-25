interface HandlerContext {
  env: Record<string, string | undefined>;
  requestId?: string;
}

interface CheckoutRequest {
  planType: 'monthly' | 'annual' | 'family' | 'gift';
  customerEmail: string;
  giftData?: {
    recipientEmail: string;
    gifterName?: string;
    message?: string;
  };
}

interface CheckoutResponse {
  sessionId: string;
  url: string;
}

type CheckoutErrorResponse = {
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

export async function stripeCheckoutHandler(
  request: Request,
  context: HandlerContext
): Promise<Response> {
  const requestId = context.requestId ?? crypto.randomUUID();

  if (request.method !== 'POST') {
    return jsonResponse<CheckoutErrorResponse>(
      { status: 'error', message: 'method_not_allowed', requestId },
      { status: 405 }
    );
  }

  const stripeKey = context.env.STRIPE_SECRET_KEY;
  if (!stripeKey) {
    return jsonResponse<CheckoutErrorResponse>(
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
    return jsonResponse<CheckoutErrorResponse>(
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

  let body: CheckoutRequest;
  try {
    body = (await request.json()) as CheckoutRequest;
  } catch {
    return jsonResponse<CheckoutErrorResponse>(
      { status: 'error', message: 'invalid_json', requestId },
      { status: 400 }
    );
  }

  // Validate email
  if (!body.customerEmail || !isValidEmail(body.customerEmail)) {
    return jsonResponse<CheckoutErrorResponse>(
      { status: 'error', message: 'invalid_email', requestId },
      { status: 422 }
    );
  }

  // Validate plan type
  if (!['monthly', 'annual', 'family', 'gift'].includes(body.planType)) {
    return jsonResponse<CheckoutErrorResponse>(
      { status: 'error', message: 'invalid_plan_type', requestId },
      { status: 422 }
    );
  }

  // Validate gift data if gift plan
  if (body.planType === 'gift') {
    if (!body.giftData?.recipientEmail || !isValidEmail(body.giftData.recipientEmail)) {
      return jsonResponse<CheckoutErrorResponse>(
        { status: 'error', message: 'invalid_recipient_email', requestId },
        { status: 422 }
      );
    }
  }

  // Get price IDs from environment
  const priceIds = {
    monthly: context.env.STRIPE_PRICE_MONTHLY,
    annual: context.env.STRIPE_PRICE_ANNUAL,
    family: context.env.STRIPE_PRICE_FAMILY,
    gift: context.env.STRIPE_PRICE_GIFT,
  };

  const siteUrl = context.env.PUBLIC_SITE_URL || 'https://curiouskelly.com';

  try {
    let sessionConfig: import('stripe').Stripe.Checkout.SessionCreateParams;

    if (body.planType === 'gift' && body.giftData) {
      // Gift purchase (one-time payment)
      sessionConfig = {
        payment_method_types: ['card'],
        line_items: [
          {
            price: priceIds.gift!,
            quantity: 1,
          },
        ],
        mode: 'payment',
        customer_email: body.customerEmail,
        success_url: `${siteUrl}/success?session_id={CHECKOUT_SESSION_ID}`,
        cancel_url: `${siteUrl}/?canceled=true`,
        metadata: {
          type: 'gift',
          recipient_email: body.giftData.recipientEmail,
          gift_message: body.giftData.message || '',
          gifter_name: body.giftData.gifterName || '',
        },
      };
    } else {
      // Subscription plans
      const priceId = priceIds[body.planType];
      if (!priceId) {
        return jsonResponse<CheckoutErrorResponse>(
          { status: 'error', message: `price_id_not_configured_${body.planType}`, requestId },
          { status: 503 }
        );
      }

      sessionConfig = {
        payment_method_types: ['card'],
        line_items: [
          {
            price: priceId,
            quantity: 1,
          },
        ],
        mode: 'subscription',
        customer_email: body.customerEmail,
        success_url: `${siteUrl}/success?session_id={CHECKOUT_SESSION_ID}`,
        cancel_url: `${siteUrl}/?canceled=true`,
        metadata: {
          type: body.planType,
        },
        subscription_data: {
          metadata: {
            plan_type: body.planType,
          },
        },
      };
    }

    const session = await stripe.checkout.sessions.create(sessionConfig);

    console.info('[stripeCheckoutHandler] session created', {
      requestId,
      sessionId: session.id,
      planType: body.planType,
      emailHash: body.customerEmail.substring(0, 3) + '***',
    });

    return jsonResponse<CheckoutResponse>({
      sessionId: session.id,
      url: session.url || '',
    });
  } catch (error) {
    console.error('[stripeCheckoutHandler] error', { requestId, error });
    return jsonResponse<CheckoutErrorResponse>(
      {
        status: 'error',
        message: 'checkout_creation_failed',
        requestId,
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}



