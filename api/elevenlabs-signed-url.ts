import type { VercelRequest, VercelResponse } from '@vercel/node';

/**
 * ElevenLabs Conversational AI Signed URL Generator
 * Creates a secure signed URL for starting conversations with Kelly
 * 
 * POST /api/elevenlabs-signed-url
 * Returns: { signedUrl: string }
 * 
 * This is required for PRIVATE agents or enhanced security.
 * For PUBLIC agents, the client can connect directly with just the agent ID.
 */
export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle preflight
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Get API key from environment
  const apiKey = process.env.ELEVENLABS_API_KEY;
  const agentId = process.env.ELEVENLABS_AGENT_ID || 'agent_3501kbg14w37er08w0mq13bvhy64';

  if (!apiKey) {
    console.error('[ElevenLabs SignedURL] ELEVENLABS_API_KEY not set');
    return res.status(500).json({ error: 'ElevenLabs API not configured' });
  }

  try {
    console.log(`[ElevenLabs SignedURL] Generating signed URL for agent: ${agentId}`);

    // Call ElevenLabs API to get signed URL
    const response = await fetch(
      `https://api.elevenlabs.io/v1/convai/conversation/get_signed_url?agent_id=${agentId}`,
      {
        method: 'GET',
        headers: {
          'xi-api-key': apiKey,
        },
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error('[ElevenLabs SignedURL] API error:', response.status, errorText);
      
      // If agent is public, return a message indicating direct connection should be used
      if (response.status === 403 || errorText.includes('public')) {
        return res.status(200).json({ 
          signedUrl: null,
          agentId: agentId,
          isPublic: true,
          message: 'Agent is public, use direct connection with agent ID'
        });
      }
      
      return res.status(response.status).json({ 
        error: 'Failed to generate signed URL',
        details: errorText 
      });
    }

    const data = await response.json();
    
    console.log('[ElevenLabs SignedURL] Successfully generated signed URL');
    
    return res.status(200).json({ 
      signedUrl: data.signed_url,
      agentId: agentId,
      isPublic: false
    });

  } catch (error) {
    console.error('[ElevenLabs SignedURL] Error:', error);
    return res.status(500).json({ 
      error: 'Internal server error',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

