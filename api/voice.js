/**
 * ElevenLabs Voice API Proxy
 * Vercel Serverless Function
 * 
 * POST /api/voice
 * Body: { text: "..." }
 * Returns: audio/mpeg stream
 */

export default async function handler(req, res) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({
      status: 'error',
      message: 'Method not allowed. Use POST.'
    });
  }

  // Get environment variables
  const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
  const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID;

  // Check if API key and voice ID are configured
  if (!ELEVENLABS_API_KEY || !ELEVENLABS_VOICE_ID) {
    console.error('[voice.js] Missing ElevenLabs configuration');
    return res.status(500).json({
      status: 'error',
      message: 'ElevenLabs API not configured. Please set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID environment variables.'
    });
  }

  // Parse request body
  let body;
  try {
    body = typeof req.body === 'string' ? JSON.parse(req.body) : req.body;
  } catch (error) {
    return res.status(400).json({
      status: 'error',
      message: 'Invalid JSON in request body'
    });
  }

  // Validate text field
  if (!body.text || typeof body.text !== 'string' || body.text.trim().length === 0) {
    return res.status(400).json({
      status: 'error',
      message: 'Missing or empty "text" field in request body'
    });
  }

  const text = body.text.trim();
  const requestId = crypto.randomUUID?.() || Date.now().toString();

  try {
    // Call ElevenLabs API
    const response = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}`,
      {
        method: 'POST',
        headers: {
          'Accept': 'audio/mpeg',
          'Content-Type': 'application/json',
          'xi-api-key': ELEVENLABS_API_KEY
        },
        body: JSON.stringify({
          text: text,
          model_id: 'eleven_multilingual_v2',
          voice_settings: {
            stability: 0.5,
            similarity_boost: 0.75,
            style: 0.5,
            use_speaker_boost: true
          }
        })
      }
    );

    // Handle ElevenLabs API errors
    if (!response.ok) {
      const errorText = await response.text();
      console.error('[voice.js] ElevenLabs API error', {
        requestId,
        status: response.status,
        error: errorText.substring(0, 200)
      });

      return res.status(response.status).json({
        status: 'error',
        message: 'ElevenLabs API error',
        details: errorText.substring(0, 200)
      });
    }

    // Get audio buffer
    const audioBuffer = await response.arrayBuffer();

    console.info('[voice.js] Voice generated successfully', {
      requestId,
      textLength: text.length,
      audioSize: audioBuffer.byteLength
    });

    // Return audio stream
    res.setHeader('Content-Type', 'audio/mpeg');
    res.setHeader('Cache-Control', 'public, max-age=3600');
    res.setHeader('Content-Length', audioBuffer.byteLength);
    res.status(200).send(Buffer.from(audioBuffer));

  } catch (error) {
    console.error('[voice.js] Voice generation error', {
      requestId,
      error: error.message || String(error)
    });

    return res.status(500).json({
      status: 'error',
      message: 'Voice generation failed',
      error: error.message || String(error)
    });
  }
}

