import { FastifyInstance } from "fastify";
import { z } from "zod";
import { prisma } from "@acme/database";
import { FeedbackPayload } from "@acme/types";
import { SessionTokenPayload } from "@acme/service-auth";

export const registerFeedbackRoutes = (app: FastifyInstance) => {
  app.route({
    method: "POST",
    url: "/v1/feedback",
    schema: {
      summary: "Submit session feedback",
      body: FeedbackPayload,
      response: {
        204: { type: "null" },
      },
    },
    preHandler: async (request) => {
      await request.jwtVerify<SessionTokenPayload>();
    },
    handler: async (request, reply) => {
      const body = FeedbackPayload.parse(request.body);
      const tokenPayload = request.user as SessionTokenPayload;

      await prisma.feedback.upsert({
        where: {
          sessionId_userId: {
            sessionId: body.sessionId,
            userId: tokenPayload.sub,
          },
        },
        update: {
          rating: body.rating,
          comment: body.comment,
          submitted: new Date(body.submittedAt),
        },
        create: {
          sessionId: body.sessionId,
          userId: tokenPayload.sub,
          rating: body.rating,
          comment: body.comment,
          submitted: new Date(body.submittedAt),
        },
      });
      reply.status(204).send();
    },
  });
};

