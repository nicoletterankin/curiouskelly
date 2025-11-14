import { rumHandler } from '../../handlers/rum';

type CloudflareContext = {
  request: Request;
  env: Record<string, string | undefined>;
};

export const onRequestPost = async (context: CloudflareContext) => {
  return rumHandler(context.request, {
    env: context.env,
    requestId: context.request.headers.get('cf-ray') ?? undefined
  });
};





