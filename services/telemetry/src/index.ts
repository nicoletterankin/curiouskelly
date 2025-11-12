import { createClient } from "@clickhouse/client";
import { env } from "@acme/config";
import { logger } from "@acme/logger";

const client = createClient({
  url: env.CLICKHOUSE_URL,
});

export interface RumEvent {
  userAgent: string;
  url: string;
  metrics: Record<string, number>;
  consent: boolean;
  capturedAt: Date;
}

export class TelemetryService {
  async ingestRum(event: RumEvent) {
    if (!env.RUM_ENABLED || !event.consent) {
      logger.debug("TelemetryService.ingestRum.skipped");
      return;
    }

    await client.insert({
      table: "rum_events",
      values: [
        {
          user_agent: event.userAgent.slice(0, 255),
          url: event.url.slice(0, 255),
          lcp: event.metrics.lcp ?? 0,
          cls: event.metrics.cls ?? 0,
          inp: event.metrics.inp ?? 0,
          captured_at: event.capturedAt.toISOString(),
        },
      ],
      format: "JSONEachRow",
    });
  }

  async ingestServerEvent(name: string, payload: Record<string, unknown>) {
    await client.insert({
      table: "server_events",
      values: [
        {
          name,
          payload: JSON.stringify(payload),
          captured_at: new Date().toISOString(),
        },
      ],
      format: "JSONEachRow",
    });
  }
}

export const telemetryService = new TelemetryService();

