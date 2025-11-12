import { FastifyInstance } from "fastify";
import { z } from "zod";
import { searchService } from "@acme/service-search";

const querySchema = z.object({
  q: z.string().min(1),
});

export const registerSearchRoutes = (app: FastifyInstance) => {
  app.route({
    method: "GET",
    url: "/v1/search",
    schema: {
      summary: "Search lesson topics",
      querystring: querySchema,
      response: {
        200: z.array(z.any()),
      },
    },
    handler: async (request, reply) => {
      const query = querySchema.parse(request.query);
      const results = await searchService.search(query.q);
      reply.send(results);
    },
  });
};

