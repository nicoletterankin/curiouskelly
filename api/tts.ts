import type { VercelRequest, VercelResponse } from '@vercel/node';

/**
 * ElevenLabs TTS Proxy API
 * Securely proxies text-to-speech requests to ElevenLabs
 * 
 * POST /api/tts
 * Body: { text: string, voiceId?: string }
 * Returns: audio/mpeg stream
 */
export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers for all responses
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  // Handle preflight OPTIONS request
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  // Only allow POST
  if (req.method !== 'POST') {
    console.log('[TTS API] Method not allowed:', req.method);
    return res.status(405).json({ error: 'Method not allowed', method: req.method });
  }

  // Get API key from environment
  const apiKey = process.env.ELEVENLABS_API_KEY;
  
  console.log('[TTS API] Request received');
  
  if (!apiKey) {
    console.error('[TTS API] ❌ ELEVENLABS_API_KEY not set in environment');
    return res.status(503).json({ 
      error: 'TTS service not configured',
      code: 'MISSING_API_KEY',
      details: 'ELEVENLABS_API_KEY environment variable is not set',
      hint: 'Add ELEVENLABS_API_KEY to Vercel Dashboard → Settings → Environment Variables',
      ttsAvailable: false
    });
  }

  // Parse request
  const { text, voiceId } = req.body;
  
  console.log('[TTS API] Request body:', { textLength: text?.length, voiceId: voiceId || 'default', hasText: !!text });
  
  if (!text || typeof text !== 'string') {
    console.error('[TTS API] ❌ Invalid text:', typeof text);
    return res.status(400).json({ 
      error: 'Text is required',
      received: typeof text,
      hint: 'Send POST with JSON body: { "text": "Hello world" }'
    });
  }

  // Default to Kelly's voice
  const voice = voiceId || process.env.ELEVENLABS_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';

  try {
    console.log(`[TTS API] 🎤 Generating audio for ${text.length} chars with voice ${voice}`);

    // Call ElevenLabs API
    const response = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${voice}`,
      {
        method: 'POST',
        headers: {
          'Accept': 'audio/mpeg',
          'Content-Type': 'application/json',
          'xi-api-key': apiKey,
        },
        body: JSON.stringify({
          text,
          model_id: 'eleven_multilingual_v2',
          voice_settings: {
            stability: 0.5,
            similarity_boost: 0.75,
            style: 0.0,
            use_speaker_boost: true,
          },
        }),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error('[TTS API] ❌ ElevenLabs API error:', response.status);
      console.error('[TTS API] Error response:', errorText);
      
      // Parse the error for more details
      let errorDetails: any = { raw: errorText };
      try {
        errorDetails = JSON.parse(errorText);
      } catch (e) {
        // Not JSON, use raw text
      }
      
      return res.status(response.status).json({ 
        error: 'TTS generation failed',
        status: response.status,
        elevenlabsError: errorDetails,
        hint: response.status === 401 ? 'API key may be invalid' :
              response.status === 429 ? 'Rate limit exceeded' :
              response.status === 400 ? 'Invalid request to ElevenLabs' :
              'Check ElevenLabs dashboard for details'
      });
    }

    // Stream audio back to client
    const audioBuffer = await response.arrayBuffer();
    
    console.log(`[TTS API] ✅ Success! Generated ${audioBuffer.byteLength} bytes of audio`);
    
    res.setHeader('Content-Type', 'audio/mpeg');
    res.setHeader('Content-Length', audioBuffer.byteLength.toString());
    // POST responses are not cached by default by CDNs; keep this conservative.
    res.setHeader('Cache-Control', 'no-store');
    
    return res.send(Buffer.from(audioBuffer));

  } catch (error) {
    console.error('[TTS API] ❌ Unexpected error:', error);
    return res.status(500).json({ 
      error: 'Internal server error',
      message: error instanceof Error ? error.message : 'Unknown error',
      stack: process.env.NODE_ENV === 'development' ? (error as Error).stack : undefined
    });
  }
}








