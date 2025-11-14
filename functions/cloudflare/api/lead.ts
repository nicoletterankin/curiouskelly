import { leadHandler } from '../../handlers/lead';

type CloudflareContext = {
  request: Request;
  env: Record<string, string | undefined>;
};

export const onRequestPost = async (context: CloudflareContext) => {
  return leadHandler(context.request, {
    env: context.env,
    requestId: context.request.headers.get('cf-ray') ?? undefined
  });
};





