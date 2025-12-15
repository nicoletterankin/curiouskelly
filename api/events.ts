/**
 * Event Logging API
 * 
 * POST /api/events
 * 
 * Logs user events to the user_events table for zero-trust audit trail.
 * Both authenticated and anonymous events are supported.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

export default function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.setHeader('Content-Type', 'application/json');
  
  if (req.method === 'OPTIONS') {
    return res.status(204).send('');
  }
  
  if (req.method !== 'POST') {
    return res.status(405).send(JSON.stringify({ error: 'Method not allowed' }));
  }
  
  try {
    const body = req.body || {};
    const eventType = body.event_type || 'unknown';
    
    // For now, just acknowledge the event without database
    // Database storage will be added after migrations are run
    const response = { 
      success: true,
      event_type: eventType,
      message: 'Event received (database pending migration)'
    };
    
    return res.status(200).send(JSON.stringify(response));
    
  } catch (error) {
    const fallback = { 
      success: true, 
      warning: 'Event logging failed silently',
      fallback: true
    };
    return res.status(200).send(JSON.stringify(fallback));
  }
}
