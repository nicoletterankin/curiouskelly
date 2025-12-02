import { stripeSessionHandler } from '../../handlers/stripe-session';

type CloudflareContext = {
  request: Request;
  env: Record<string, string | undefined>;
};

export const onRequestGet = async (context: CloudflareContext) => {
  return stripeSessionHandler(context.request, {
    env: context.env,
    requestId: context.request.headers.get('cf-ray') ?? undefined
  });
};











