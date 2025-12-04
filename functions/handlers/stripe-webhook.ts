/**
 * Stripe Webhook Handler
 * Processes Stripe events for subscriptions, payments, and customer lifecycle
 */

interface HandlerContext {
  env: Record<string, string | undefined>;
  requestId?: string;
}

interface WebhookResponse {
  received: boolean;
  event?: string;
  error?: string;
}

function jsonResponse<T>(body: T, init?: ResponseInit) {
  return new Response(JSON.stringify(body), {
    status: init?.status ?? 200,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      ...(init?.headers ?? {})
    }
  });
}

export async function stripeWebhookHandler(
  request: Request,
  context: HandlerContext
): Promise<Response> {
  const requestId = context.requestId ?? crypto.randomUUID();

  if (request.method !== 'POST') {
    return jsonResponse<WebhookResponse>(
      { received: false, error: 'method_not_allowed' },
      { status: 405 }
    );
  }

  const stripeKey = context.env.STRIPE_SECRET_KEY;
  const webhookSecret = context.env.STRIPE_WEBHOOK_SECRET;
  
  if (!stripeKey || !webhookSecret) {
    console.error('[stripeWebhook] Missing configuration', { requestId });
    return jsonResponse<WebhookResponse>(
      { received: false, error: 'webhook_not_configured' },
      { status: 503 }
    );
  }

  // Dynamic import Stripe
  let Stripe: typeof import('stripe').default;
  try {
    const stripeModule = await import('stripe');
    Stripe = stripeModule.default;
  } catch (error) {
    console.error('[stripeWebhook] Failed to load Stripe', { requestId, error });
    return jsonResponse<WebhookResponse>(
      { received: false, error: 'stripe_module_unavailable' },
      { status: 503 }
    );
  }

  const stripe = new Stripe(stripeKey, {
    apiVersion: '2024-11-20.acacia',
  });

  // Get the raw body and signature
  const body = await request.text();
  const signature = request.headers.get('stripe-signature');

  if (!signature) {
    console.error('[stripeWebhook] Missing signature', { requestId });
    return jsonResponse<WebhookResponse>(
      { received: false, error: 'missing_signature' },
      { status: 400 }
    );
  }

  // Verify webhook signature
  let event: import('stripe').Stripe.Event;
  try {
    event = stripe.webhooks.constructEvent(body, signature, webhookSecret);
  } catch (err) {
    console.error('[stripeWebhook] Signature verification failed', { requestId, error: err });
    return jsonResponse<WebhookResponse>(
      { received: false, error: 'invalid_signature' },
      { status: 400 }
    );
  }

  console.info('[stripeWebhook] Event received', {
    requestId,
    eventId: event.id,
    eventType: event.type,
  });

  // Handle the event
  try {
    switch (event.type) {
      case 'checkout.session.completed':
        await handleCheckoutComplete(event.data.object, context);
        break;

      case 'customer.subscription.created':
        await handleSubscriptionCreated(event.data.object, context);
        break;

      case 'customer.subscription.updated':
        await handleSubscriptionUpdated(event.data.object, context);
        break;

      case 'customer.subscription.deleted':
        await handleSubscriptionDeleted(event.data.object, context);
        break;

      case 'invoice.paid':
        await handleInvoicePaid(event.data.object, context);
        break;

      case 'invoice.payment_failed':
        await handleInvoicePaymentFailed(event.data.object, context);
        break;

      case 'charge.refunded':
        await handleChargeRefunded(event.data.object, context);
        break;

      case 'charge.dispute.created':
        await handleDisputeCreated(event.data.object, context);
        break;

      default:
        console.info('[stripeWebhook] Unhandled event type', { eventType: event.type });
    }

    return jsonResponse<WebhookResponse>({
      received: true,
      event: event.type,
    });

  } catch (error) {
    console.error('[stripeWebhook] Handler error', { requestId, eventType: event.type, error });
    return jsonResponse<WebhookResponse>(
      { received: false, error: 'handler_error' },
      { status: 500 }
    );
  }
}

// Event Handlers

async function handleCheckoutComplete(
  session: import('stripe').Stripe.Checkout.Session,
  context: HandlerContext
) {
  console.info('[handleCheckoutComplete]', {
    sessionId: session.id,
    customerEmail: session.customer_email,
    mode: session.mode,
    metadata: session.metadata,
    clientReferenceId: session.client_reference_id,
  });

  const metadata = session.metadata || {};
  const customerEmail = session.customer_email;

  if (!customerEmail) {
    console.warn('[handleCheckoutComplete] No customer email');
    return;
  }

  // Track affiliate attribution - check both metadata and client_reference_id
  // client_reference_id format from Payment Links: aff_CODE_timestamp
  let affiliateCode = metadata.affiliate_code;
  
  if (!affiliateCode && session.client_reference_id) {
    const refId = session.client_reference_id;
    if (refId.startsWith('aff_')) {
      affiliateCode = refId; // Pass the full string, trackAffiliateConversion will parse it
      console.info('[handleCheckoutComplete] Found affiliate code in client_reference_id', { refId });
    }
  }

  if (affiliateCode) {
    await trackAffiliateConversion(affiliateCode, session, context);
  }

  // Handle gift purchases
  if (metadata.type === 'gift' && metadata.recipient_email) {
    await createGiftCode(session, metadata, context);
    return;
  }

  // Handle lifetime purchases
  if (metadata.type === 'lifetime') {
    await grantLifetimeAccess(customerEmail, session, context);
    return;
  }

  // Subscription access is handled by subscription.created webhook
  console.info('[handleCheckoutComplete] Subscription checkout - waiting for subscription.created');
}

async function handleSubscriptionCreated(
  subscription: import('stripe').Stripe.Subscription,
  context: HandlerContext
) {
  console.info('[handleSubscriptionCreated]', {
    subscriptionId: subscription.id,
    customerId: subscription.customer,
    status: subscription.status,
  });

  // Get customer email
  // In production, you'd fetch from Stripe or your database
  const metadata = subscription.metadata || {};
  
  // Grant access based on subscription
  // TODO: Integrate with Supabase to update user record
  console.info('[handleSubscriptionCreated] Grant access', {
    subscriptionId: subscription.id,
    planType: metadata.plan_type,
  });
}

async function handleSubscriptionUpdated(
  subscription: import('stripe').Stripe.Subscription,
  context: HandlerContext
) {
  console.info('[handleSubscriptionUpdated]', {
    subscriptionId: subscription.id,
    status: subscription.status,
    cancelAtPeriodEnd: subscription.cancel_at_period_end,
  });

  // Handle plan changes, cancellation scheduling, etc.
  if (subscription.cancel_at_period_end) {
    console.info('[handleSubscriptionUpdated] Subscription scheduled for cancellation');
    // TODO: Send retention email
  }
}

async function handleSubscriptionDeleted(
  subscription: import('stripe').Stripe.Subscription,
  context: HandlerContext
) {
  console.info('[handleSubscriptionDeleted]', {
    subscriptionId: subscription.id,
    customerId: subscription.customer,
  });

  // Revoke access
  // TODO: Update user record in Supabase
  
  // Trigger win-back email sequence
  // TODO: Add to email queue
  console.info('[handleSubscriptionDeleted] Trigger win-back sequence');
}

async function handleInvoicePaid(
  invoice: import('stripe').Stripe.Invoice,
  context: HandlerContext
) {
  console.info('[handleInvoicePaid]', {
    invoiceId: invoice.id,
    customerId: invoice.customer,
    amountPaid: invoice.amount_paid,
  });

  // Record payment for analytics
  // Extend access if needed
}

async function handleInvoicePaymentFailed(
  invoice: import('stripe').Stripe.Invoice,
  context: HandlerContext
) {
  console.info('[handleInvoicePaymentFailed]', {
    invoiceId: invoice.id,
    customerId: invoice.customer,
    attemptCount: invoice.attempt_count,
  });

  // Send dunning email
  // TODO: Integrate with email service
  console.info('[handleInvoicePaymentFailed] Send dunning email');
}

async function handleChargeRefunded(
  charge: import('stripe').Stripe.Charge,
  context: HandlerContext
) {
  console.info('[handleChargeRefunded]', {
    chargeId: charge.id,
    customerId: charge.customer,
    amountRefunded: charge.amount_refunded,
  });

  // Update access if full refund
  if (charge.refunded) {
    console.info('[handleChargeRefunded] Full refund - revoke access');
    // TODO: Update user record
  }
}

async function handleDisputeCreated(
  dispute: import('stripe').Stripe.Dispute,
  context: HandlerContext
) {
  console.error('[handleDisputeCreated] ALERT: Dispute created', {
    disputeId: dispute.id,
    chargeId: dispute.charge,
    amount: dispute.amount,
    reason: dispute.reason,
  });

  // Alert team immediately
  // TODO: Send Slack/email notification
}

// Helper Functions

async function trackAffiliateConversion(
  affiliateCode: string,
  session: import('stripe').Stripe.Checkout.Session,
  context: HandlerContext
) {
  // Extract affiliate code from client_reference_id (format: aff_CODE_timestamp)
  let code = affiliateCode;
  if (affiliateCode.startsWith('aff_')) {
    code = affiliateCode.split('_')[1];
  }

  console.info('[trackAffiliateConversion]', {
    originalCode: affiliateCode,
    parsedCode: code,
    sessionId: session.id,
    amountTotal: session.amount_total,
  });

  const supabaseUrl = context.env.PUBLIC_SUPABASE_URL;
  const supabaseKey = context.env.SUPABASE_SERVICE_KEY || context.env.PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseKey) {
    console.warn('[trackAffiliateConversion] Missing Supabase config');
    return;
  }

  try {
    // Find affiliate by referral code
    const affiliateResponse = await fetch(
      `${supabaseUrl}/rest/v1/affiliates?referral_code=eq.${code}&select=id,commission_rate,tier`,
      {
        headers: {
          'apikey': supabaseKey,
          'Authorization': `Bearer ${supabaseKey}`,
        },
      }
    );

    const affiliates = await affiliateResponse.json();
    
    if (!affiliates || affiliates.length === 0) {
      console.warn('[trackAffiliateConversion] Affiliate not found', { code });
      return;
    }

    const affiliate = affiliates[0];
    const subscriptionValue = (session.amount_total || 0) / 100; // Convert from cents
    const commissionRate = affiliate.commission_rate || 20;
    const commissionEarned = subscriptionValue * (commissionRate / 100);

    console.info('[trackAffiliateConversion] Creating referral record', {
      affiliateId: affiliate.id,
      subscriptionValue,
      commissionRate,
      commissionEarned,
    });

    // Create referral record
    const referralResponse = await fetch(
      `${supabaseUrl}/rest/v1/referrals`,
      {
        method: 'POST',
        headers: {
          'apikey': supabaseKey,
          'Authorization': `Bearer ${supabaseKey}`,
          'Content-Type': 'application/json',
          'Prefer': 'return=representation',
        },
        body: JSON.stringify({
          affiliate_id: affiliate.id,
          referred_user_id: session.client_reference_id || session.id, // Use session ID as fallback
          referral_code: code,
          subscription_value: subscriptionValue,
          commission_earned: commissionEarned,
          status: 'confirmed',
        }),
      }
    );

    if (!referralResponse.ok) {
      const error = await referralResponse.text();
      console.error('[trackAffiliateConversion] Failed to create referral', { error });
      return;
    }

    // Update affiliate stats (increment referral count)
    const updateResponse = await fetch(
      `${supabaseUrl}/rest/v1/affiliates?id=eq.${affiliate.id}`,
      {
        method: 'PATCH',
        headers: {
          'apikey': supabaseKey,
          'Authorization': `Bearer ${supabaseKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          total_referrals: affiliate.total_referrals ? affiliate.total_referrals + 1 : 1,
          active_referrals: affiliate.active_referrals ? affiliate.active_referrals + 1 : 1,
          lifetime_earnings: affiliate.lifetime_earnings 
            ? parseFloat(affiliate.lifetime_earnings) + commissionEarned 
            : commissionEarned,
        }),
      }
    );

    if (!updateResponse.ok) {
      console.error('[trackAffiliateConversion] Failed to update affiliate stats');
    }

    console.info('[trackAffiliateConversion] Successfully tracked conversion', {
      affiliateId: affiliate.id,
      code,
      commissionEarned,
    });

  } catch (error) {
    console.error('[trackAffiliateConversion] Error', { error });
  }
}

async function createGiftCode(
  session: import('stripe').Stripe.Checkout.Session,
  metadata: Record<string, string>,
  context: HandlerContext
) {
  const giftCode = generateGiftCode();
  
  console.info('[createGiftCode]', {
    sessionId: session.id,
    giftCode,
    recipientEmail: metadata.recipient_email,
    deliveryDate: metadata.delivery_date,
  });

  // TODO: Store gift code in database
  // TODO: Schedule delivery email for delivery_date
  // TODO: Send confirmation to purchaser
}

async function grantLifetimeAccess(
  email: string,
  session: import('stripe').Stripe.Checkout.Session,
  context: HandlerContext
) {
  console.info('[grantLifetimeAccess]', {
    email,
    sessionId: session.id,
  });

  // TODO: Update user record with lifetime flag
  // TODO: Send welcome email
}

function generateGiftCode(): string {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  let code = 'GIFT-';
  for (let i = 0; i < 8; i++) {
    code += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return code;
}

