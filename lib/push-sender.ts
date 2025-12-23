/**
 * Curious Kelly Push Notification Sender
 * 
 * Unified push notification sending for:
 * - Web Push (VAPID)
 * - iOS (APNs) 
 * - Android (FCM)
 * 
 * Environment Variables Required:
 * - VAPID_PUBLIC_KEY: Web push public key
 * - VAPID_PRIVATE_KEY: Web push private key
 * - VAPID_SUBJECT: mailto:hello@curiouskelly.com
 * - FIREBASE_SERVER_KEY: For Android FCM (optional, can use newer FCM v1)
 * - APNS_KEY_ID: Apple Push Notification key ID
 * - APNS_TEAM_ID: Apple Developer Team ID
 * - APNS_AUTH_KEY: APNs auth key (.p8 file contents)
 */

import webpush from 'web-push';
import { createClient } from '@supabase/supabase-js';
import type { SupabaseClient } from '@supabase/supabase-js';

type AnySupabaseClient = SupabaseClient<any, any, any, any, any>;

// Types
export interface PushPayload {
  title: string;
  body: string;
  icon?: string;
  badge?: string;
  url?: string;
  tag?: string;
  data?: Record<string, unknown>;
}

export interface PushToken {
  id: string;
  user_id: string;
  device_token: string;
  platform: 'ios' | 'android' | 'web' | 'macos' | 'windows' | 'linux';
  is_active: boolean;
}

export interface SendResult {
  success: boolean;
  tokenId: string;
  platform: string;
  error?: string;
}

// Configure web-push with VAPID keys
const VAPID_PUBLIC_KEY = process.env.VAPID_PUBLIC_KEY;
const VAPID_PRIVATE_KEY = process.env.VAPID_PRIVATE_KEY;
const VAPID_SUBJECT = process.env.VAPID_SUBJECT || 'mailto:hello@curiouskelly.com';

if (VAPID_PUBLIC_KEY && VAPID_PRIVATE_KEY) {
  webpush.setVapidDetails(VAPID_SUBJECT, VAPID_PUBLIC_KEY, VAPID_PRIVATE_KEY);
}

/**
 * Send a web push notification
 */
export async function sendWebPush(
  endpoint: string,
  p256dh: string,
  auth: string,
  payload: PushPayload
): Promise<SendResult> {
  if (!VAPID_PUBLIC_KEY || !VAPID_PRIVATE_KEY) {
    return {
      success: false,
      tokenId: '',
      platform: 'web',
      error: 'VAPID keys not configured'
    };
  }

  try {
    const subscription = {
      endpoint,
      keys: { p256dh, auth }
    };

    const pushPayload = JSON.stringify({
      title: payload.title,
      body: payload.body,
      icon: payload.icon || '/images/kelly/kelly-icon.png',
      badge: payload.badge || '/images/kelly/kelly-badge.png',
      url: payload.url || '/kelly.html',
      tag: payload.tag || 'kelly-notification',
      data: payload.data
    });

    await webpush.sendNotification(subscription, pushPayload, {
      TTL: 86400, // 24 hours
      urgency: 'normal'
    });

    return { success: true, tokenId: endpoint, platform: 'web' };
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    
    // Handle expired subscriptions (410 Gone)
    if (errorMessage.includes('410') || errorMessage.includes('expired')) {
      return {
        success: false,
        tokenId: endpoint,
        platform: 'web',
        error: 'subscription_expired'
      };
    }

    return {
      success: false,
      tokenId: endpoint,
      platform: 'web',
      error: errorMessage
    };
  }
}

/**
 * Send an iOS push notification via APNs
 * Note: Requires APNs authentication key (.p8) from Apple Developer account
 */
export async function sendAPNs(
  deviceToken: string,
  payload: PushPayload
): Promise<SendResult> {
  const APNS_KEY_ID = process.env.APNS_KEY_ID;
  const APNS_TEAM_ID = process.env.APNS_TEAM_ID;
  const APNS_AUTH_KEY = process.env.APNS_AUTH_KEY;
  const APNS_BUNDLE_ID = process.env.APNS_BUNDLE_ID || 'com.curiouskelly.app';

  if (!APNS_KEY_ID || !APNS_TEAM_ID || !APNS_AUTH_KEY) {
    return {
      success: false,
      tokenId: deviceToken,
      platform: 'ios',
      error: 'APNs credentials not configured'
    };
  }

  try {
    // Generate JWT for APNs
    const jwt = await generateAPNsJWT(APNS_KEY_ID, APNS_TEAM_ID, APNS_AUTH_KEY);
    
    const apnsPayload = {
      aps: {
        alert: {
          title: payload.title,
          body: payload.body
        },
        badge: 1,
        sound: 'default',
        'mutable-content': 1
      },
      url: payload.url || 'https://curiouskelly.com',
      data: payload.data
    };

    // Use production APNs for App Store builds
    const apnsHost = process.env.NODE_ENV === 'production' 
      ? 'api.push.apple.com' 
      : 'api.sandbox.push.apple.com';

    const response = await fetch(`https://${apnsHost}/3/device/${deviceToken}`, {
      method: 'POST',
      headers: {
        'authorization': `bearer ${jwt}`,
        'apns-topic': APNS_BUNDLE_ID,
        'apns-push-type': 'alert',
        'apns-priority': '10',
        'apns-expiration': '0',
        'content-type': 'application/json'
      },
      body: JSON.stringify(apnsPayload)
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(`APNs error ${response.status}: ${errorBody}`);
    }

    return { success: true, tokenId: deviceToken, platform: 'ios' };
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    
    // Handle invalid tokens
    if (errorMessage.includes('BadDeviceToken') || errorMessage.includes('Unregistered')) {
      return {
        success: false,
        tokenId: deviceToken,
        platform: 'ios',
        error: 'token_invalid'
      };
    }

    return {
      success: false,
      tokenId: deviceToken,
      platform: 'ios',
      error: errorMessage
    };
  }
}

/**
 * Generate JWT for APNs authentication
 */
async function generateAPNsJWT(
  keyId: string,
  teamId: string,
  authKey: string
): Promise<string> {
  // APNs JWT expires in 1 hour
  const now = Math.floor(Date.now() / 1000);
  
  const header = {
    alg: 'ES256',
    kid: keyId
  };
  
  const payload = {
    iss: teamId,
    iat: now
  };

  // In production, use a proper JWT library like jose
  // For now, we'll use a simple implementation
  const headerB64 = Buffer.from(JSON.stringify(header)).toString('base64url');
  const payloadB64 = Buffer.from(JSON.stringify(payload)).toString('base64url');
  
  // Note: This is a placeholder - actual ES256 signing requires the crypto module
  // In production, use the 'jose' or 'jsonwebtoken' package
  const crypto = await import('crypto');
  const sign = crypto.createSign('SHA256');
  sign.update(`${headerB64}.${payloadB64}`);
  const signature = sign.sign(authKey, 'base64url');
  
  return `${headerB64}.${payloadB64}.${signature}`;
}

/**
 * Send an Android push notification via FCM
 * Note: Requires Firebase Admin SDK or FCM v1 API credentials
 */
export async function sendFCM(
  deviceToken: string,
  payload: PushPayload
): Promise<SendResult> {
  const FIREBASE_PROJECT_ID = process.env.FIREBASE_PROJECT_ID;
  const FIREBASE_PRIVATE_KEY = process.env.FIREBASE_PRIVATE_KEY;
  const FIREBASE_CLIENT_EMAIL = process.env.FIREBASE_CLIENT_EMAIL;

  // Legacy FCM key (simpler but deprecated)
  const FIREBASE_SERVER_KEY = process.env.FIREBASE_SERVER_KEY;

  if (FIREBASE_SERVER_KEY) {
    // Use legacy FCM API (simpler)
    return sendFCMLegacy(deviceToken, payload, FIREBASE_SERVER_KEY);
  }

  if (!FIREBASE_PROJECT_ID || !FIREBASE_PRIVATE_KEY || !FIREBASE_CLIENT_EMAIL) {
    return {
      success: false,
      tokenId: deviceToken,
      platform: 'android',
      error: 'FCM credentials not configured'
    };
  }

  try {
    // Get OAuth2 access token for FCM v1 API
    const accessToken = await getFirebaseAccessToken(
      FIREBASE_CLIENT_EMAIL,
      FIREBASE_PRIVATE_KEY
    );

    const fcmPayload = {
      message: {
        token: deviceToken,
        notification: {
          title: payload.title,
          body: payload.body
        },
        android: {
          notification: {
            icon: 'ic_notification',
            color: '#2563eb',
            click_action: 'OPEN_ACTIVITY',
            channel_id: 'kelly_daily'
          }
        },
        data: {
          url: payload.url || 'https://curiouskelly.com',
          ...Object.fromEntries(
            Object.entries(payload.data || {}).map(([k, v]) => [k, String(v)])
          )
        }
      }
    };

    const response = await fetch(
      `https://fcm.googleapis.com/v1/projects/${FIREBASE_PROJECT_ID}/messages:send`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(fcmPayload)
      }
    );

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(`FCM error ${response.status}: ${errorBody}`);
    }

    return { success: true, tokenId: deviceToken, platform: 'android' };
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    
    // Handle invalid tokens
    if (errorMessage.includes('UNREGISTERED') || errorMessage.includes('NOT_FOUND')) {
      return {
        success: false,
        tokenId: deviceToken,
        platform: 'android',
        error: 'token_invalid'
      };
    }

    return {
      success: false,
      tokenId: deviceToken,
      platform: 'android',
      error: errorMessage
    };
  }
}

/**
 * Send FCM using legacy API (simpler setup)
 */
async function sendFCMLegacy(
  deviceToken: string,
  payload: PushPayload,
  serverKey: string
): Promise<SendResult> {
  try {
    const fcmPayload = {
      to: deviceToken,
      notification: {
        title: payload.title,
        body: payload.body,
        icon: 'ic_notification',
        color: '#2563eb',
        click_action: 'FCM_PLUGIN_ACTIVITY'
      },
      data: {
        url: payload.url || 'https://curiouskelly.com',
        ...(payload.data || {})
      }
    };

    const response = await fetch('https://fcm.googleapis.com/fcm/send', {
      method: 'POST',
      headers: {
        'Authorization': `key=${serverKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(fcmPayload)
    });

    const result = await response.json();

    if (result.failure > 0) {
      const error = result.results?.[0]?.error || 'Unknown FCM error';
      if (error === 'NotRegistered' || error === 'InvalidRegistration') {
        return {
          success: false,
          tokenId: deviceToken,
          platform: 'android',
          error: 'token_invalid'
        };
      }
      throw new Error(error);
    }

    return { success: true, tokenId: deviceToken, platform: 'android' };
  } catch (error: unknown) {
    return {
      success: false,
      tokenId: deviceToken,
      platform: 'android',
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}

/**
 * Get Firebase access token for FCM v1 API
 */
async function getFirebaseAccessToken(
  clientEmail: string,
  privateKey: string
): Promise<string> {
  const now = Math.floor(Date.now() / 1000);
  
  const header = { alg: 'RS256', typ: 'JWT' };
  const payload = {
    iss: clientEmail,
    scope: 'https://www.googleapis.com/auth/firebase.messaging',
    aud: 'https://oauth2.googleapis.com/token',
    iat: now,
    exp: now + 3600
  };

  const headerB64 = Buffer.from(JSON.stringify(header)).toString('base64url');
  const payloadB64 = Buffer.from(JSON.stringify(payload)).toString('base64url');
  
  const crypto = await import('crypto');
  const sign = crypto.createSign('RSA-SHA256');
  sign.update(`${headerB64}.${payloadB64}`);
  const signature = sign.sign(privateKey, 'base64url');
  
  const jwt = `${headerB64}.${payloadB64}.${signature}`;

  const response = await fetch('https://oauth2.googleapis.com/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: `grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer&assertion=${jwt}`
  });

  const data = await response.json();
  return data.access_token;
}

/**
 * Send push notification to a specific token based on platform
 */
export async function sendPushNotification(
  token: PushToken,
  payload: PushPayload,
  supabase?: AnySupabaseClient
): Promise<SendResult> {
  let result: SendResult;

  switch (token.platform) {
    case 'web':
      // Web push tokens are stored as JSON with endpoint, p256dh, auth
      try {
        const webToken = JSON.parse(token.device_token);
        result = await sendWebPush(
          webToken.endpoint,
          webToken.p256dh,
          webToken.auth,
          payload
        );
      } catch {
        result = {
          success: false,
          tokenId: token.id,
          platform: 'web',
          error: 'Invalid web push token format'
        };
      }
      break;

    case 'ios':
    case 'macos':
      result = await sendAPNs(token.device_token, payload);
      break;

    case 'android':
      result = await sendFCM(token.device_token, payload);
      break;

    case 'windows':
    case 'linux':
      // Desktop apps use web push via Electron
      try {
        const desktopToken = JSON.parse(token.device_token);
        result = await sendWebPush(
          desktopToken.endpoint,
          desktopToken.p256dh,
          desktopToken.auth,
          payload
        );
      } catch {
        result = {
          success: false,
          tokenId: token.id,
          platform: token.platform,
          error: 'Invalid desktop push token format'
        };
      }
      break;

    default:
      result = {
        success: false,
        tokenId: token.id,
        platform: token.platform,
        error: `Unsupported platform: ${token.platform}`
      };
  }

  // Update token status if invalid
  if (supabase && (result.error === 'token_invalid' || result.error === 'subscription_expired')) {
    const sb: any = supabase as any;
    await sb
      .from('push_tokens')
      .update({
        is_active: false,
        updated_at: new Date().toISOString()
      })
      .eq('id', token.id);
  }

  // Increment failed count if error
  if (supabase && !result.success && result.error !== 'token_invalid') {
    const sb: any = supabase as any;
    await sb
      .from('push_tokens')
      .update({
        failed_count: token.is_active ? 1 : undefined, // Will be incremented in actual impl
        updated_at: new Date().toISOString()
      })
      .eq('id', token.id);
  }

  return result;
}

/**
 * Send push notification to all of a user's active devices
 */
export async function sendToUser(
  userId: string,
  payload: PushPayload,
  supabase: AnySupabaseClient
): Promise<{ sent: number; failed: number; results: SendResult[] }> {
  // Get all active tokens for user
  const { data: tokens, error } = await supabase
    .from('push_tokens')
    .select('*')
    .eq('user_id', userId)
    .eq('is_active', true);

  if (error || !tokens?.length) {
    return { sent: 0, failed: 0, results: [] };
  }

  const results = await Promise.all(
    tokens.map(token => sendPushNotification(token as PushToken, payload, supabase))
  );

  const sent = results.filter(r => r.success).length;
  const failed = results.filter(r => !r.success).length;

  return { sent, failed, results };
}

/**
 * Send push notification to multiple users
 */
export async function sendToUsers(
  userIds: string[],
  payload: PushPayload,
  supabase: AnySupabaseClient
): Promise<{ totalSent: number; totalFailed: number; userResults: Map<string, SendResult[]> }> {
  const userResults = new Map<string, SendResult[]>();
  let totalSent = 0;
  let totalFailed = 0;

  // Process in batches of 100
  const batchSize = 100;
  for (let i = 0; i < userIds.length; i += batchSize) {
    const batch = userIds.slice(i, i + batchSize);
    
    const results = await Promise.all(
      batch.map(async (userId) => {
        const result = await sendToUser(userId, payload, supabase);
        userResults.set(userId, result.results);
        return result;
      })
    );

    results.forEach(r => {
      totalSent += r.sent;
      totalFailed += r.failed;
    });

    // Rate limit: small delay between batches
    if (i + batchSize < userIds.length) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }

  return { totalSent, totalFailed, userResults };
}



