import { FastifyInstance } from "fastify";
import { createHmac, timingSafeEqual } from "node:crypto";
import { paymentsService } from "@acme/service-payments";
import { classroomService } from "@acme/service-classroom";
import { logger } from "@acme/logger";
import { z } from "zod";
import { env } from "@acme/config";

const livekitEventSchema = z.object({
  event: z.string(),
  room: z.object({
    name: z.string(),
  }),
  participant: z
    .object({
      identity: z.string().optional(),
    })
    .optional(),
});

export const registerWebhookRoutes = (app: FastifyInstance) => {
  app.route({
    method: "POST",
    url: "/webhooks/stripe",
    config: {
      rawBody: true,
    },
    schema: {
      hide: true,
    },
    handler: async (request, reply) => {
      const signatureHeader = request.headers["stripe-signature"];
      if (!signatureHeader || Array.isArray(signatureHeader)) {
        reply.code(400).send({ error: "Missing signature" });
        return;
      }

      const rawBody = request.rawBody as string | Buffer | undefined;
      if (!rawBody) {
        reply.code(400).send({ error: "Missing payload" });
        return;
      }

      await paymentsService.handleWebhook(
        signatureHeader,
        Buffer.isBuffer(rawBody) ? rawBody : Buffer.from(rawBody),
      );
      reply.status(204).send();
    },
  });

  app.route({
    method: "POST",
    url: "/webhooks/livekit",
    config: {
      rawBody: true,
    },
    schema: {
      hide: true,
    },
    handler: async (request, reply) => {
      const signatureHeader = request.headers["x-livekit-signature"];
      if (!signatureHeader || Array.isArray(signatureHeader)) {
        reply.code(401).send({ error: "Missing signature" });
        return;
      }

      const rawBody = request.rawBody as string | Buffer | undefined;
      if (!rawBody) {
        reply.code(400).send({ error: "Missing payload" });
        return;
      }

      const payloadBuffer = Buffer.isBuffer(rawBody) ? rawBody : Buffer.from(rawBody);
      const expected = createHmac("sha256", env.LIVEKIT_WEBHOOK_SECRET).update(payloadBuffer).digest();
      const provided = Buffer.from(signatureHeader, "hex");

      if (provided.length !== expected.length || !timingSafeEqual(provided, expected)) {
        reply.code(401).send({ error: "Invalid signature" });
        return;
      }

      const body = livekitEventSchema.parse(JSON.parse(payloadBuffer.toString("utf8")));
      logger.info({ event: body.event }, "LiveKit webhook received");

      if (body.event === "room_ended") {
        await classroomService.closeSessionByRoom(body.room.name);
      }

      if (body.event === "participant_left" && body.participant?.identity) {
        await classroomService.markParticipantLeftByRoom(body.room.name, body.participant.identity);
      }

      reply.status(204).send();
    },
  });
};

