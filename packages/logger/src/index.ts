import pino, { LoggerOptions, TransportMultiOptions } from "pino";
import { context, trace } from "@opentelemetry/api";
import { env } from "@acme/config";

type RedactableValue = string | number | boolean | null | undefined;

const SENSITIVE_KEYS = ["password", "secret", "token", "key", "authorization", "cookie"];

const redactValue = (value: RedactableValue): RedactableValue => {
  if (value === null || value === undefined) {
    return value;
  }
  if (typeof value === "string" && value.length > 4) {
    return `${value.slice(0, 2)}***${value.slice(-2)}`;
  }
  return "***";
};

const scrubObject = (data: Record<string, unknown>): Record<string, unknown> => {
  return Object.fromEntries(
    Object.entries(data).map(([key, value]) => {
      if (SENSITIVE_KEYS.some((sensitive) => key.toLowerCase().includes(sensitive))) {
        return [key, redactValue(value as RedactableValue)];
      }
      if (value && typeof value === "object" && !Array.isArray(value)) {
        return [key, scrubObject(value as Record<string, unknown>)];
      }
      return [key, value];
    }),
  );
};

const baseOptions: LoggerOptions = {
  level: env.LOG_LEVEL,
  timestamp: pino.stdTimeFunctions.isoTime,
  formatters: {
    bindings(bindings) {
      return {
        pid: bindings.pid,
        host: bindings.hostname,
        component: bindings.name,
      };
    },
    log(object) {
      const span = trace.getSpan(context.active());
      const spanContext = span?.spanContext();
      return {
        ...scrubObject(object),
        traceId: spanContext?.traceId,
        spanId: spanContext?.spanId,
      };
    },
  },
  redact: {
    paths: ["req.headers.authorization", "req.headers.cookie", "res.headers.set-cookie"],
    remove: true,
  },
};

const transport: TransportMultiOptions | undefined =
  env.NODE_ENV === "development"
    ? {
        targets: [
          {
            level: env.LOG_LEVEL,
            target: "pino-pretty",
            options: {
              colorize: true,
              translateTime: "SYS:standard",
              singleLine: false,
            },
          },
        ],
      }
    : undefined;

export const logger = pino({
  ...baseOptions,
  transport,
});

export type Logger = typeof logger;


