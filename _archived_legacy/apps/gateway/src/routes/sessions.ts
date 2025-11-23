import { FastifyInstance } from "fastify";
import { z } from "zod";
import { scheduleService } from "@acme/service-schedule";
import { classroomService } from "@acme/service-classroom";
import { entitlementsService } from "@acme/service-entitlements";
import { telemetryService } from "@acme/service-telemetry";
import { prisma } from "@acme/database";
import { JoinResponse, ApiError } from "@acme/types";
import { SessionTokenPayload } from "@acme/service-auth";

const joinBody = z.object({
  slotId: z.string().uuid(),
});

const attendanceBody = z.object({
  status: z.enum(["join", "leave"]),
  timestamp: z.string().datetime(),
});

export const registerSessionRoutes = (app: FastifyInstance) => {
  app.route({
    method: "POST",
    url: "/v1/join",
    schema: {
      summary: "Join an hourly class session",
      body: joinBody,
      response: {
        200: JoinResponse,
        401: ApiError,
        409: ApiError,
      },
    },
    preHandler: async (request) => {
      await request.jwtVerify<SessionTokenPayload>();
    },
    handler: async (request, reply) => {
      const body = joinBody.parse(request.body);
      const tokenPayload = request.user as SessionTokenPayload;

      const entitlements = await entitlementsService.getFeaturesForUser(tokenPayload.sub);
      const planLimits = entitlements.features;

      const todayCount = await prisma.attendance.count({
        where: {
          userId: tokenPayload.sub,
          joinedAt: {
            gte: new Date(new Date().setHours(0, 0, 0, 0)),
          },
        },
      });

      if (
        typeof planLimits.maxInteractivePerDay === "number" &&
        todayCount >= planLimits.maxInteractivePerDay
      ) {
        reply.code(409).send({ error: "Daily interactive limit reached", code: "LIMIT_REACHED" });
        return;
      }

      const slot = await prisma.scheduleSlot.findUnique({
        where: { id: body.slotId },
      });
      if (!slot) {
        reply.code(409).send({ error: "Slot not available", code: "SLOT_UNAVAILABLE" });
        return;
      }

      const reserved = await scheduleService.reserveSeat(body.slotId, tokenPayload.sub);
      if (!reserved && !slot.allowOverflow) {
        reply.code(409).send({ error: "Session full", code: "SESSION_FULL" });
        return;
      }

      const session = await classroomService.ensureSessionForSlot(body.slotId, tokenPayload.sub);

      const basePolicy = {
        maxDevices: Number(planLimits.maxConcurrentDevices ?? 1),
        maxInteractivePerDay: Number(planLimits.maxInteractivePerDay ?? 0),
        features: Object.keys(planLimits),
      };

      const response =
        slot.allowOverflow && !reserved
          ? {
              mode: "spectator" as const,
              spectator: await classroomService.createOverflowSpectatorAccess(session.id),
              policy: basePolicy,
              roomHint: session.livekitRoomId,
            }
          : (() => {
              return classroomService
                .mintAccessToken(session.id, tokenPayload.sub, tokenPayload.sub, {
                  plan: entitlements.planCode,
                })
                .then((interactive) => ({
                  mode: "interactive" as const,
                  lobbyToken: interactive.token,
                  policy: basePolicy,
                  roomHint: session.livekitRoomId,
                }));
            })();

      const payload = await response;

      await telemetryService.ingestServerEvent("session_join", {
        userId: tokenPayload.sub,
        sessionId: session.id,
        slotId: body.slotId,
        overflow: !reserved,
      });

      reply.send(payload);
    },
  });

  app.route({
    method: "POST",
    url: "/v1/sessions/:id/attendance",
    schema: {
      summary: "Mark session attendance events",
      params: z.object({
        id: z.string().uuid(),
      }),
      body: attendanceBody,
      response: {
        204: { type: "null" },
      },
    },
    preHandler: async (request) => {
      await request.jwtVerify<SessionTokenPayload>();
    },
    handler: async (request, reply) => {
      const params = z.object({ id: z.string().uuid() }).parse(request.params);
      const body = attendanceBody.parse(request.body);
      const tokenPayload = request.user as SessionTokenPayload;

      if (body.status === "leave") {
        await classroomService.markParticipantLeft(params.id, tokenPayload.sub);
      } else {
        await prisma.attendance.upsert({
          where: {
            sessionId_userId: {
              sessionId: params.id,
              userId: tokenPayload.sub,
            },
          },
          update: {
            joinedAt: new Date(body.timestamp),
          },
          create: {
            sessionId: params.id,
            userId: tokenPayload.sub,
            joinedAt: new Date(body.timestamp),
          },
        });
      }
      reply.status(204).send();
    },
  });

  app.route({
    method: "GET",
    url: "/v1/sessions/:id/replay",
    schema: {
      summary: "Retrieve replay URL for a paid session",
      params: z.object({
        id: z.string().uuid(),
      }),
      response: {
        200: z.object({
          replayUrl: z.string().url(),
        }),
        404: ApiError,
      },
    },
    preHandler: async (request) => {
      await request.jwtVerify<SessionTokenPayload>();
    },
    handler: async (request, reply) => {
      const params = z.object({ id: z.string().uuid() }).parse(request.params);
      const replayUrl = await classroomService.getReplayUrl(params.id);
      if (!replayUrl) {
        reply.code(404).send({ error: "Replay unavailable", code: "REPLAY_NOT_FOUND" });
        return;
      }

      reply.send({ replayUrl });
    },
  });
};

