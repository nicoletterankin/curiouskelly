import type { VercelRequest, VercelResponse } from '@vercel/node';
import { configHandler } from '../../handlers/config';

function createRequest(req: VercelRequest) {
  const protocol =
    (req.headers['x-forwarded-proto'] as string | undefined) ??
    (req.headers['x-vercel-proto'] as string | undefined) ??
    'https';
  const url = `${protocol}://${req.headers.host}${req.url}`;
  return new Request(url, {
    method: req.method,
    headers: req.headers as Record<string, string>,
  });
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const request = createRequest(req);
  const response = await configHandler(request, {
    env: process.env,
  });

  // Convert Response to Vercel response
  const body = await response.text();
  res.status(response.status);
  response.headers.forEach((value, key) => {
    res.setHeader(key, value);
  });
  res.send(body);
}

