/**
 * /api/time.ts - Kelly Time Authority
 * 
 * Server-authoritative time endpoint for global time synchronization.
 * All clients sync to this clock to ensure everyone sees the same time.
 * 
 * "When Kelly says 9:00:00 AM, class starts. Everywhere. For everyone. To the second."
 */

export const config = {
  runtime: 'edge',
};

export default function handler(req: Request): Response {
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
  
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: {
      'Content-Type': 'application/json',
      // CRITICAL: Never cache time responses
      'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
      'Pragma': 'no-cache',
      'Expires': '0',
      // Allow cross-origin for widget embeds
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}


