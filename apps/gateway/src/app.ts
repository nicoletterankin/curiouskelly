import Fastify from "fastify";
import cookie from "@fastify/cookie";
import jwt from "@fastify/jwt";
import rateLimit from "@fastify/rate-limit";
import swagger from "@fastify/swagger";
import swaggerUi from "@fastify/swagger-ui";
import fastifyRawBody from "fastify-raw-body";
import { serializerCompiler, validatorCompiler } from "fastify-type-provider-zod";
import { env } from "@acme/config";
import { logger } from "@acme/logger";
import { registerLessonRoutes } from "./routes/lessons.js";
import { registerScheduleRoutes } from "./routes/schedule.js";
import { registerSessionRoutes } from "./routes/sessions.js";
import { registerSearchRoutes } from "./routes/search.js";
import { registerFeedbackRoutes } from "./routes/feedback.js";
import { registerWebhookRoutes } from "./routes/webhooks.js";

export const buildServer = () => {
  const app = Fastify({
    logger,
    trustProxy: true,
  }).withTypeProvider<{ serializer: typeof serializerCompiler; validator: typeof validatorCompiler }>();

  app.setValidatorCompiler(validatorCompiler);
  app.setSerializerCompiler(serializerCompiler);

  app.register(cookie, {
    parseOptions: {},
  });

  app.register(fastifyRawBody, {
    field: "rawBody",
    global: false,
    encoding: "utf8",
    runFirst: true,
  });

  app.register(jwt, {
    secret: env.JWT_SECRET,
    cookie: {
      cookieName: "session",
      signed: false,
    },
  });

  app.register(rateLimit, {
    max: 100,
    timeWindow: "1 minute",
    keyGenerator: (req) => req.ip,
  });

  app.register(swagger, {
    openapi: {
      info: {
        title: "Daily Lesson Platform API",
        version: "1.0.0",
      },
      servers: [
        {
          url: "https://api.example.com",
        },
      ],
    },
  });

  app.register(swaggerUi, {
    routePrefix: "/docs",
  });

  app.get(
    "/health",
    {
      schema: {
        response: {
          200: {
            type: "object",
            properties: {
              status: { type: "string" },
            },
          },
        },
      },
    },
    async () => ({ status: "ok" }),
  );

  registerLessonRoutes(app);
  registerScheduleRoutes(app);
  registerSessionRoutes(app);
  registerSearchRoutes(app);
  registerFeedbackRoutes(app);
  registerWebhookRoutes(app);

  return app;
};

