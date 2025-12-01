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
  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Get API key from environment
  const apiKey = process.env.ELEVENLABS_API_KEY;
  if (!apiKey) {
    console.error('[TTS API] ELEVENLABS_API_KEY not set in environment');
    return res.status(500).json({ error: 'TTS service not configured' });
  }

  // Parse request
  const { text, voiceId } = req.body;
  
  if (!text || typeof text !== 'string') {
    return res.status(400).json({ error: 'Text is required' });
  }

  // Default to Kelly's voice
  const voice = voiceId || process.env.ELEVENLABS_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';

  try {
    console.log(`[TTS API] Generating audio for ${text.length} chars with voice ${voice}`);

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
          model_id: 'eleven_monolingual_v1',
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
      console.error('[TTS API] ElevenLabs error:', response.status, errorText);
      return res.status(response.status).json({ 
        error: 'TTS generation failed',
        details: errorText 
      });
    }

    // Stream audio back to client
    const audioBuffer = await response.arrayBuffer();
    
    res.setHeader('Content-Type', 'audio/mpeg');
    res.setHeader('Content-Length', audioBuffer.byteLength.toString());
    res.setHeader('Cache-Control', 'public, max-age=31536000'); // Cache for 1 year
    
    return res.send(Buffer.from(audioBuffer));

  } catch (error) {
    console.error('[TTS API] Error:', error);
    return res.status(500).json({ 
      error: 'Internal server error',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}






