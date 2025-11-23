import { FastifyInstance } from "fastify";
import { z } from "zod";
import { lessonsService } from "@acme/service-lessons";
import { ApiError, LessonManifest } from "@acme/types";

const querySchema = z.object({
  topic: z.string().min(1),
  locale: z.string().default("en-US"),
});

export const registerLessonRoutes = (app: FastifyInstance) => {
  app.route({
    method: "GET",
    url: "/v1/lessons/today",
    schema: {
      summary: "Get today's lesson manifest",
      querystring: querySchema,
      response: {
        200: LessonManifest,
        404: ApiError,
      },
    },
    handler: async (request, reply) => {
      const query = querySchema.parse(request.query);
      const manifest = await lessonsService.getTodayManifest(query.topic, query.locale);
      if (!manifest) {
        reply.code(404).send({ error: "Lesson not found", code: "LESSON_NOT_FOUND" });
        return;
      }
      reply.send(manifest);
    },
  });
};

