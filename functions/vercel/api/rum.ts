import { Buffer } from 'node:buffer';
import type { VercelRequest, VercelResponse } from '@vercel/node';
import { rumHandler } from '../../handlers/rum';

function createRequest(req: VercelRequest) {
  const protocol =
    (req.headers['x-forwarded-proto'] as string | undefined) ??
    (req.headers['x-vercel-proto'] as string | undefined) ??
    'https';
  const url = `${protocol}://${req.headers.host}${req.url}`;
  const body =
    req.method && ['GET', 'HEAD'].includes(req.method)
      ? undefined
      : typeof req.body === 'string'
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
  const response = await rumHandler(request, {
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

