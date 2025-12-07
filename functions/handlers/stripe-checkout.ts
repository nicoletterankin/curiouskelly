interface HandlerContext {
  env: Record<string, string | undefined>;
  requestId?: string;
}

interface CheckoutRequest {
  planType: 'monthly' | 'annual' | 'lifetime' | 'gift_12mo' | 'gift_6mo' | 'gift_3mo';
  customerEmail: string;
  promoCode?: string;
  affiliateCode?: string;
  giftData?: {
    recipientEmail: string;
    gifterName?: string;
    message?: string;
    deliveryDate?: string;
  };
  // Attribution
  utmSource?: string;
  utmMedium?: string;
  utmCampaign?: string;
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
    apiVersion: '2024-11-20.acacia'
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
  const validPlanTypes = ['monthly', 'annual', 'lifetime', 'gift_12mo', 'gift_6mo', 'gift_3mo'];
  if (!validPlanTypes.includes(body.planType)) {
    return jsonResponse<CheckoutErrorResponse>(
      { status: 'error', message: 'invalid_plan_type', requestId },
      { status: 422 }
    );
  }

  // Validate gift data if gift plan
  const isGiftPlan = body.planType.startsWith('gift_');
  if (isGiftPlan) {
    if (!body.giftData?.recipientEmail || !isValidEmail(body.giftData.recipientEmail)) {
      return jsonResponse<CheckoutErrorResponse>(
        { status: 'error', message: 'invalid_recipient_email', requestId },
        { status: 422 }
      );
    }
  }

  // Get price IDs from environment
  const priceIds: Record<string, string | undefined> = {
    monthly: context.env.STRIPE_PRICE_MONTHLY,
    annual: context.env.STRIPE_PRICE_ANNUAL,
    lifetime: context.env.STRIPE_PRICE_LIFETIME,
    gift_12mo: context.env.STRIPE_PRICE_GIFT_12MO,
    gift_6mo: context.env.STRIPE_PRICE_GIFT_6MO,
    gift_3mo: context.env.STRIPE_PRICE_GIFT_3MO
  };

  const siteUrl = context.env.PUBLIC_SITE_URL || 'https://curiouskelly.com';

  // Build common metadata for attribution tracking
  const commonMetadata = {
    source: 'web',
    utm_source: body.utmSource || 'direct',
    utm_medium: body.utmMedium || 'none',
    utm_campaign: body.utmCampaign || 'none',
    affiliate_code: body.affiliateCode || '',
    promo_code: body.promoCode || ''
  };

  try {
    let sessionConfig: import('stripe').Stripe.Checkout.SessionCreateParams;

    if (isGiftPlan && body.giftData) {
      // Gift purchase (one-time payment)
      const giftPriceId = priceIds[body.planType];
      if (!giftPriceId) {
        return jsonResponse<CheckoutErrorResponse>(
          { status: 'error', message: `price_id_not_configured_${body.planType}`, requestId },
          { status: 503 }
        );
      }

      sessionConfig = {
        payment_method_types: ['card'],
        line_items: [
          {
            price: giftPriceId,
            quantity: 1
          }
        ],
        mode: 'payment',
        customer_email: body.customerEmail,
        success_url: `${siteUrl}/gift-success.html?session_id={CHECKOUT_SESSION_ID}&plan=${body.planType}`,
        cancel_url: `${siteUrl}/payment-cancelled.html`,
        allow_promotion_codes: true,
        metadata: {
          ...commonMetadata,
          type: 'gift',
          gift_plan: body.planType,
          recipient_email: body.giftData.recipientEmail,
          gift_message: body.giftData.message || '',
          gifter_name: body.giftData.gifterName || '',
          delivery_date: body.giftData.deliveryDate || new Date().toISOString()
        }
      };
    } else if (body.planType === 'lifetime') {
      // Lifetime purchase (one-time payment)
      const lifetimePriceId = priceIds.lifetime;
      if (!lifetimePriceId) {
        return jsonResponse<CheckoutErrorResponse>(
          { status: 'error', message: 'price_id_not_configured_lifetime', requestId },
          { status: 503 }
        );
      }

      sessionConfig = {
        payment_method_types: ['card'],
        line_items: [
          {
            price: lifetimePriceId,
            quantity: 1
          }
        ],
        mode: 'payment',
        customer_email: body.customerEmail,
        success_url: `${siteUrl}/welcome.html?session_id={CHECKOUT_SESSION_ID}&plan=lifetime`,
        cancel_url: `${siteUrl}/payment-cancelled.html`,
        allow_promotion_codes: true,
        metadata: {
          ...commonMetadata,
          type: 'lifetime'
        }
      };
    } else {
      // Subscription plans (monthly, annual)
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
            quantity: 1
          }
        ],
        mode: 'subscription',
        customer_email: body.customerEmail,
        success_url: `${siteUrl}/welcome.html?session_id={CHECKOUT_SESSION_ID}`,
        cancel_url: `${siteUrl}/payment-cancelled.html`,
        metadata: {
          ...commonMetadata,
          type: body.planType
        },
        subscription_data: {
          metadata: {
            ...commonMetadata,
            plan_type: body.planType
          }
        },
        // Allow promotion codes
        allow_promotion_codes: true,
        // Collect billing address for tax purposes
        billing_address_collection: 'auto',
        // Enable automatic tax calculation
        automatic_tax: { enabled: true }
      };
    }

    const session = await stripe.checkout.sessions.create(sessionConfig);

    console.info('[stripeCheckoutHandler] session created', {
      requestId,
      sessionId: session.id,
      planType: body.planType,
      emailHash: body.customerEmail.substring(0, 3) + '***'
    });

    return jsonResponse<CheckoutResponse>({
      sessionId: session.id,
      url: session.url || ''
    });
  } catch (error) {
    console.error('[stripeCheckoutHandler] error', { requestId, error });
    return jsonResponse<CheckoutErrorResponse>(
      {
        status: 'error',
        message: 'checkout_creation_failed',
        requestId,
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}
