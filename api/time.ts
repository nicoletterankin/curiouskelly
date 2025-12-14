/**
 * /api/time.ts - Kelly Time Authority
 * 
 * Server-authoritative time endpoint for global time synchronization.
 * All clients sync to this clock to ensure everyone sees the same time.
 * 
 * "When Kelly says 9:00:00 AM, class starts. Everywhere. For everyone. To the second."
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

export default function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0');
    return res.status(204).send('');
  }

  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const now = Date.now();
  const date = new Date(now);
  
  // Response payload with multiple time formats
  const payload = {
    // Millisecond timestamp (primary for sync calculations)
    utc: now,
    
    // ISO 8601 format
    iso: date.toISOString(),
    
    // Unix timestamp (seconds)
    unix: Math.floor(now / 1000),
    
    // Pre-formatted strings for display (UTC)
    formatted: {
      date: date.toISOString().split('T')[0],
      time: date.toISOString().split('T')[1].split('.')[0],
    },
    
    // Server info
    server: 'kelly-time-authority',
    version: '1.0.0',
  };

  // CRITICAL: Never cache time responses
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0');
  res.setHeader('Pragma', 'no-cache');
  res.setHeader('Expires', '0');
  // Allow cross-origin for widget embeds
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  return res.status(200).send(JSON.stringify(payload));
}



