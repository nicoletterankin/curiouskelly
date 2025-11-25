import { Buffer } from 'node:buffer';
import { leadHandler } from '../handlers/lead';

// Netlify function event type (fallback if @netlify/functions not available)
type NetlifyHandlerEvent = {
  httpMethod: string;
  body: string | null;
  isBase64Encoded: boolean;
  rawUrl: string;
  headers: Record<string, string>;
};

type NetlifyHandlerContext = {
  env: Record<string, string>;
};

type NetlifyHandler = (event: NetlifyHandlerEvent, context: NetlifyHandlerContext) => Promise<{
  statusCode: number;
  headers: Record<string, string>;
  body: string;
}>;

function buildRequest(event: NetlifyHandlerEvent) {
  const body =
    event.httpMethod && ['GET', 'HEAD'].includes(event.httpMethod)
      ? undefined
      : event.body && event.isBase64Encoded
        ? Buffer.from(event.body, 'base64').toString()
        : event.body ?? undefined;

  return new Request(event.rawUrl, {
    method: event.httpMethod,
    headers: event.headers as Record<string, string>,
    body
  });
}

export const handler: NetlifyHandler = async (event, context) => {
  const request = buildRequest(event);
  const response = await leadHandler(request, {
    env: {
      ...process.env,
      ...context.env
    },
    requestId: event.headers['x-nf-request-id']
  });

  const headers: Record<string, string> = {};
  response.headers.forEach((value, key) => {
    headers[key] = value;
  });
  const body = await response.text();

  return {
    statusCode: response.status,
    headers,
    body
  };
};

