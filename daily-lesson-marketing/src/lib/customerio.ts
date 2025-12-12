/**
 * Customer.io Integration
 * Email automation and transactional emails
 * 
 * Account: "Lesson of the Day, PBC"
 * Site ID: 9ea8fc826910bbd745a3
 */

interface CustomerProfile {
  id: string;
  email: string;
  first_name?: string;
  created_at?: number;
  trial_end_date?: string;
  plan?: 'monthly' | 'annual' | 'gift';
}

interface EventData {
  name: string;
  data?: Record<string, any>;
}

/**
 * Track API - For identifying users and tracking events
 */
const TRACK_API_URL = 'https://track.customer.io/api/v1';
const siteId = import.meta.env.CUSTOMER_IO_SITE_ID;
const apiKey = import.meta.env.CUSTOMER_IO_API_KEY;

// Check if Customer.io is configured
const isConfigured = Boolean(siteId && apiKey);

// Create basic auth header (only if configured)
const authHeader = isConfigured ? `Basic ${btoa(`${siteId}:${apiKey}`)}` : '';

/**
 * Identify a user/customer in Customer.io
 * This creates or updates a customer profile
 */
export async function identifyCustomer(profile: CustomerProfile): Promise<boolean> {
  if (!isConfigured) {
    console.warn('Customer.io not configured - skipping identify');
    return false;
  }
  
  try {
    const response = await fetch(`${TRACK_API_URL}/customers/${profile.id}`, {
      method: 'PUT',
      headers: {
        'Authorization': authHeader,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(profile),
    });

    return response.ok;
  } catch (error) {
    console.error('Customer.io identify error:', error);
    return false;
  }
}

/**
 * Track an event for a customer
 * Events trigger automated campaigns in Customer.io
 */
export async function trackEvent(
  customerId: string,
  event: EventData
): Promise<boolean> {
  if (!isConfigured) {
    console.warn('Customer.io not configured - skipping track event:', event.name);
    return false;
  }
  
  try {
    const response = await fetch(`${TRACK_API_URL}/customers/${customerId}/events`, {
      method: 'POST',
      headers: {
        'Authorization': authHeader,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        name: event.name,
        data: event.data || {},
      }),
    });

    return response.ok;
  } catch (error) {
    console.error('Customer.io track event error:', error);
    return false;
  }
}

/**
 * Send a transactional email
 * Use this for immediate one-off emails (not automated campaigns)
 */
export async function sendTransactionalEmail(options: {
  transactionalMessageId: string;
  to: string;
  identifiers: { id?: string; email?: string };
  messageData?: Record<string, any>;
}): Promise<boolean> {
  const appApiKey = import.meta.env.CUSTOMER_IO_APP_API_KEY;
  
  if (!appApiKey) {
    console.warn('Customer.io App API Key not configured - skipping transactional email');
    return false;
  }
  
  try {
    const response = await fetch('https://api.customer.io/v1/send/email', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${appApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        transactional_message_id: options.transactionalMessageId,
        to: options.to,
        identifiers: options.identifiers,
        message_data: options.messageData || {},
      }),
    });

    return response.ok;
  } catch (error) {
    console.error('Customer.io send email error:', error);
    return false;
  }
}

/**
 * Helper: Track trial started
 */
export async function trackTrialStarted(customerId: string, data: {
  email: string;
  first_name: string;
  trial_end_date: string;
  plan: 'monthly' | 'annual';
}): Promise<boolean> {
  // First, identify the customer
  await identifyCustomer({
    id: customerId,
    email: data.email,
    first_name: data.first_name,
    created_at: Math.floor(Date.now() / 1000),
    trial_end_date: data.trial_end_date,
    plan: data.plan,
  });

  // Then track the event
  return trackEvent(customerId, {
    name: 'trial_started',
    data: {
      trial_end_date: data.trial_end_date,
      plan: data.plan,
    },
  });
}

/**
 * Helper: Track subscription purchased
 */
export async function trackSubscriptionPurchased(customerId: string, data: {
  plan: 'monthly' | 'annual' | 'gift';
  amount: number;
  stripe_customer_id?: string;
  stripe_subscription_id?: string;
}): Promise<boolean> {
  return trackEvent(customerId, {
    name: 'subscription_purchased',
    data,
  });
}

/**
 * Helper: Track lesson completed
 */
export async function trackLessonCompleted(customerId: string, data: {
  lesson_id: string;
  lesson_title: string;
  streak: number;
  total_lessons: number;
}): Promise<boolean> {
  return trackEvent(customerId, {
    name: 'lesson_completed',
    data,
  });
}

/**
 * Helper: Track payment failed
 */
export async function trackPaymentFailed(customerId: string, data: {
  amount: number;
  reason?: string;
  stripe_invoice_id?: string;
}): Promise<boolean> {
  return trackEvent(customerId, {
    name: 'payment_failed',
    data,
  });
}





