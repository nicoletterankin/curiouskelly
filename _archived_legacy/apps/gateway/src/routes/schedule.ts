import { FastifyInstance } from "fastify";
import { z } from "zod";
import { scheduleService } from "@acme/service-schedule";
import { ApiError, Region } from "@acme/types";

const querySchema = z.object({
  region: Region,
  topic: z.string().min(1),
});

const nextSlotsResponse = z.array(
  z.object({
    id: z.string().uuid(),
    topic: z.string(),
    region: Region,
    startTime: z.string(),
    capacity: z.number(),
    instructorId: z.string().uuid(),
    allowOverflow: z.boolean(),
  }),
);

export const registerScheduleRoutes = (app: FastifyInstance) => {
  app.route({
    method: "GET",
    url: "/v1/schedule/next",
    schema: {
      summary: "Next hourly slots",
      querystring: querySchema,
      response: {
        200: nextSlotsResponse,
        404: ApiError,
      },
    },
    handler: async (request, reply) => {
      const query = querySchema.parse(request.query);
      const slots = await scheduleService.getNextSlots({
        topic: query.topic,
        region: query.region,
      });
      if (!slots.length) {
        reply.code(404).send({ error: "No slots found", code: "SLOTS_NOT_FOUND" });
        return;
      }
      reply.send(slots);
    },
  });
};

