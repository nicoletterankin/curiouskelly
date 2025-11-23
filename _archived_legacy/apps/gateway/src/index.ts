import { buildServer } from "./app.js";
import { paymentsService } from "@acme/service-payments";

const start = async () => {
  const server = buildServer();
  try {
    await paymentsService.startDunningWorker();
    await server.listen({ port: Number(process.env.PORT ?? 4000), host: "0.0.0.0" });
    server.log.info("Gateway listening");
  } catch (error) {
    server.log.error(error);
    process.exit(1);
  }
};

start();

