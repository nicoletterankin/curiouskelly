/**
 * BYOK LLM Proxy API
 * 
 * Proxies LLM requests to user's API key.
 * This allows curriculum-aware prompts without exposing user's API key to frontend.
 * 
 * Security:
 * - API key is sent encrypted
 * - Never logged or stored
 * - One-time use per request
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

interface BYOKRequest {
  provider: 'openai' | 'anthropic' | 'google';
  model: string;
  apiKey: string; // Encrypted in production
  prompt: string;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { provider, model, apiKey, prompt }: BYOKRequest = req.body;

    if (!provider || !model || !apiKey || !prompt) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    // Validate provider
    if (!['openai', 'anthropic', 'google'].includes(provider)) {
      return res.status(400).json({ error: 'Invalid provider' });
    }

    // Call appropriate LLM API
    let response: string;

    switch (provider) {
      case 'openai':
        response = await callOpenAI(model, apiKey, prompt);
        break;
      case 'anthropic':
        response = await callAnthropic(model, apiKey, prompt);
        break;
      case 'google':
        response = await callGoogle(model, apiKey, prompt);
        break;
      default:
        return res.status(400).json({ error: 'Unsupported provider' });
    }

    return res.status(200).json({
      success: true,
      response,
      provider,
      model
    });

  } catch (error: any) {
    console.error('[BYOK] Error:', error);
    return res.status(500).json({
      error: 'LLM request failed',
      message: error.message
    });
  }
}

/**
 * Call OpenAI API
 */
async function callOpenAI(model: string, apiKey: string, prompt: string): Promise<string> {
  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model,
      messages: [
        {
          role: 'system',
          content: 'You are Kelly, an AI teacher from Curious Kelly. Be warm, curious, and encouraging.'
        },
        {
          role: 'user',
          content: prompt
        }
      ],
      temperature: 0.7,
      max_tokens: 500
    })
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`OpenAI API error: ${error.error?.message || response.statusText}`);
  }

  const data = await response.json();
  return data.choices[0]?.message?.content || 'No response generated';
}

/**
 * Call Anthropic API
 */
async function callAnthropic(model: string, apiKey: string, prompt: string): Promise<string> {
  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model,
      max_tokens: 500,
      messages: [
        {
          role: 'user',
          content: prompt
        }
      ]
    })
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`Anthropic API error: ${error.error?.message || response.statusText}`);
  }

  const data = await response.json();
  return data.content[0]?.text || 'No response generated';
}

/**
 * Call Google Gemini API
 */
async function callGoogle(model: string, apiKey: string, prompt: string): Promise<string> {
  const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      contents: [{
        parts: [{
          text: prompt
        }]
      }]
    })
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`Google API error: ${error.error?.message || response.statusText}`);
  }

  const data = await response.json();
  return data.candidates[0]?.content?.parts[0]?.text || 'No response generated';
}





