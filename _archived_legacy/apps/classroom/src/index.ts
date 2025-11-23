import Fastify from "fastify";
import websocket from "@fastify/websocket";
import { logger } from "@acme/logger";
import { classroomService } from "@acme/service-classroom";

const app = Fastify({ logger });

app.register(websocket);

app.get("/health", async () => ({ status: "ok" }));

app.register(async (instance) => {
  instance.get(
    "/v1/classroom/:sessionId",
    { websocket: true },
    (connection, req) => {
      const { sessionId } = req.params as { sessionId: string };

      connection.socket.on("message", async (raw) => {
        try {
          const event = JSON.parse(raw.toString());
          await classroomService.recordEvent(event);
        } catch (error) {
          logger.error({ error }, "classroom.websocket.message_failed");
        }
      });

      connection.socket.on("close", () => {
        logger.info({ sessionId }, "classroom.websocket.closed");
      });
    },
  );
});

const start = async () => {
  try {
    await app.listen({ port: Number(process.env.PORT ?? 4100), host: "0.0.0.0" });
    logger.info("Classroom control plane listening");
  } catch (error) {
    logger.error(error);
    process.exit(1);
  }
};

start();

