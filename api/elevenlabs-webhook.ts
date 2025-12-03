import type { VercelRequest, VercelResponse } from '@vercel/node';

/**
 * ElevenLabs Conversational AI Webhook Handler
 * Receives real-time events from ElevenLabs during conversations
 * 
 * POST /api/elevenlabs-webhook
 * 
 * Events:
 * - conversation.started
 * - conversation.ended
 * - agent.response (transcript + audio metadata)
 * - user.transcript (what user said)
 * - agent.thinking (before response)
 */

interface ElevenLabsWebhookPayload {
  type: string;
  conversation_id?: string;
  timestamp?: string;
  data?: {
    transcript?: string;
    audio_duration?: number;
    visemes?: Array<{ time: number; viseme: string }>;
    user_transcript?: string;
    agent_response?: string;
  };
}

// Store for broadcasting to connected clients (in production, use Redis/Pusher)
const conversationEvents: Map<string, ElevenLabsWebhookPayload[]> = new Map();

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle preflight
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  // GET: Retrieve events for a conversation (for polling)
  if (req.method === 'GET') {
    const conversationId = req.query.conversation_id as string;
    if (conversationId) {
      const events = conversationEvents.get(conversationId) || [];
      return res.status(200).json({ events });
    }
    return res.status(400).json({ error: 'conversation_id required' });
  }

  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const payload = req.body as ElevenLabsWebhookPayload;
    
    console.log('[ElevenLabs Webhook] Received:', payload.type, {
      conversationId: payload.conversation_id,
      timestamp: payload.timestamp
    });

    // Store event for the conversation
    if (payload.conversation_id) {
      if (!conversationEvents.has(payload.conversation_id)) {
        conversationEvents.set(payload.conversation_id, []);
      }
      conversationEvents.get(payload.conversation_id)!.push(payload);
      
      // Cleanup old events (keep last 100 per conversation)
      const events = conversationEvents.get(payload.conversation_id)!;
      if (events.length > 100) {
        events.splice(0, events.length - 100);
      }
    }

    // Process specific event types
    switch (payload.type) {
      case 'conversation.started':
        console.log('[ElevenLabs Webhook] Conversation started:', payload.conversation_id);
        // Could trigger analytics, initialize session, etc.
        break;

      case 'conversation.ended':
        console.log('[ElevenLabs Webhook] Conversation ended:', payload.conversation_id);
        // Cleanup, save transcript, analytics
        // Remove stored events after delay
        setTimeout(() => {
          conversationEvents.delete(payload.conversation_id!);
        }, 60000); // Keep for 1 minute after end
        break;

      case 'agent.response':
        console.log('[ElevenLabs Webhook] Agent response:', {
          transcript: payload.data?.transcript?.substring(0, 50) + '...',
          hasVisemes: !!payload.data?.visemes,
          audioDuration: payload.data?.audio_duration
        });
        // This is where we could extract viseme data for lip-sync
        // and broadcast to connected Unity/Web clients
        break;

      case 'user.transcript':
        console.log('[ElevenLabs Webhook] User said:', payload.data?.user_transcript);
        // Could update UI, log for analytics, etc.
        break;

      case 'agent.thinking':
        console.log('[ElevenLabs Webhook] Agent thinking...');
        // Could trigger "Kelly is thinking" state in UI
        break;

      default:
        console.log('[ElevenLabs Webhook] Unknown event type:', payload.type);
    }

    return res.status(200).json({ 
      received: true,
      type: payload.type,
      conversation_id: payload.conversation_id
    });

  } catch (error) {
    console.error('[ElevenLabs Webhook] Error:', error);
    return res.status(500).json({ 
      error: 'Internal server error',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

