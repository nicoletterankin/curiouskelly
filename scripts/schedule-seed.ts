import { scheduleService } from "@acme/service-schedule";
import { env } from "@acme/config";
import { logger } from "@acme/logger";

const topics = ["daily-lesson"];
const regions: Array<"AMER" | "EUROPE" | "APAC"> = ["AMER", "EUROPE", "APAC"];

const run = async () => {
  logger.info("Seeding schedule horizon");
  await Promise.all(
    regions.flatMap((region) =>
      topics.map((topic) => scheduleService.seedMissingSlots(region, topic)),
    ),
  );
  logger.info({ env: env.NODE_ENV }, "Schedule seed complete");
  process.exit(0);
};

run().catch((error) => {
  logger.error(error);
  process.exit(1);
});

