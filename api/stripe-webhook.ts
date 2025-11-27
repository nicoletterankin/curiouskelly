import { Buffer } from 'node:buffer';
import type { VercelRequest, VercelResponse } from '@vercel/node';
import { stripeWebhookHandler } from '../functions/handlers/stripe-webhook';

function createRequest(req: VercelRequest) {
  const protocol =
    (req.headers['x-forwarded-proto'] as string | undefined) ??
    (req.headers['x-vercel-proto'] as string | undefined) ??
    'https';
  const url = `${protocol}://${req.headers.host}${req.url}`;
  
  // For webhooks, we need the raw body
  const body = typeof req.body === 'string' 
    ? req.body 
    : JSON.stringify(req.body ?? {});
  
  return new Request(url, {
    method: req.method,
    headers: req.headers as Record<string, string>,
    body
  });
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const request = createRequest(req);
  const response = await stripeWebhookHandler(request, {
    env: process.env,
    requestId: (req.headers['x-request-id'] as string | undefined) ?? undefined
  });

  res.status(response.status);
  response.headers.forEach((value, key) => {
    res.setHeader(key, value);
  });
  const buffer = Buffer.from(await response.arrayBuffer());
  res.send(buffer);
}

// Disable body parsing - we need raw body for signature verification
export const config = {
  api: {
    bodyParser: false,
  },
};

