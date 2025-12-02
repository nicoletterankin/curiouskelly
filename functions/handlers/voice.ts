interface HandlerContext {
  env: Record<string, string | undefined>;
  requestId?: string;
}

interface VoiceRequest {
  text: string;
}

function jsonResponse<T>(body: T, init?: ResponseInit) {
  return new Response(JSON.stringify(body), {
    status: init?.status ?? 200,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'no-store',
      ...(init?.headers ?? {})
    }
  });
}

export async function voiceHandler(
  request: Request,
  context: HandlerContext
): Promise<Response> {
  const requestId = context.requestId ?? crypto.randomUUID();

  if (request.method !== 'POST') {
    return jsonResponse(
      { status: 'error', message: 'method_not_allowed', requestId },
      { status: 405 }
    );
  }

  const ELEVENLABS_API_KEY = context.env.ELEVENLABS_API_KEY;
  const ELEVENLABS_VOICE_ID = context.env.ELEVENLABS_VOICE_ID;

  if (!ELEVENLABS_API_KEY || !ELEVENLABS_VOICE_ID) {
    return jsonResponse(
      { status: 'error', message: 'elevenlabs_not_configured', requestId },
      { status: 500 }
    );
  }

  let body: VoiceRequest;
  try {
    body = (await request.json()) as VoiceRequest;
  } catch {
    return jsonResponse(
      { status: 'error', message: 'invalid_json', requestId },
      { status: 400 }
    );
  }

  if (!body.text || typeof body.text !== 'string' || body.text.trim().length === 0) {
    return jsonResponse(
      { status: 'error', message: 'text_required', requestId },
      { status: 400 }
    );
  }

  try {
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
          text: body.text.trim(),
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

    if (!response.ok) {
      const errorText = await response.text();
      console.error('[voiceHandler] ElevenLabs API error', {
        requestId,
        status: response.status,
        error: errorText.substring(0, 200)
      });
      return jsonResponse(
        { status: 'error', message: 'elevenlabs_api_error', requestId },
        { status: response.status }
      );
    }

    const audioBuffer = await response.arrayBuffer();

    console.info('[voiceHandler] Voice generated successfully', {
      requestId,
      textLength: body.text.length,
      audioSize: audioBuffer.byteLength
    });

    return new Response(audioBuffer, {
      status: 200,
      headers: {
        'Content-Type': 'audio/mpeg',
        'Cache-Control': 'public, max-age=3600',
        'Content-Length': audioBuffer.byteLength.toString()
      }
    });

  } catch (error) {
    console.error('[voiceHandler] Voice generation error', {
      requestId,
      error: error instanceof Error ? error.message : String(error)
    });
    return jsonResponse(
      { status: 'error', message: 'voice_generation_failed', requestId },
      { status: 500 }
    );
  }
}









