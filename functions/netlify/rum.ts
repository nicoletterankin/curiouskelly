import { Buffer } from 'node:buffer';
import type { Handler } from '@netlify/functions';
import { rumHandler } from '../handlers/rum';

function buildRequest(event: Parameters<Handler>[0]) {
  const body =
    event.httpMethod && ['GET', 'HEAD'].includes(event.httpMethod)
      ? undefined
      : event.body && event.isBase64Encoded
        ? Buffer.from(event.body, 'base64')
        : event.body ?? undefined;

  return new Request(event.rawUrl, {
    method: event.httpMethod,
    headers: event.headers as Record<string, string>,
    body
  });
}

export const handler: Handler = async (event, context) => {
  const request = buildRequest(event);
  const response = await rumHandler(request, {
    env: {
      ...process.env,
      ...context.env
    },
    requestId: event.headers['x-nf-request-id']
  });

  const headers = Object.fromEntries(response.headers.entries());
  const body = await response.text();

  return {
    statusCode: response.status,
    headers,
    body
  };
};

