import { PrismaClient } from "@prisma/client";
import { env } from "@acme/config";
import { logger } from "@acme/logger";

declare global {
  // eslint-disable-next-line no-var
  var __prismaClient: PrismaClient | undefined;
}

export const prisma: PrismaClient =
  global.__prismaClient ??
  new PrismaClient({
    datasources: {
      db: {
        url: env.DATABASE_URL,
      },
    },
    log:
      env.NODE_ENV === "development"
        ? [
            { emit: "event", level: "query" },
            { emit: "stdout", level: "error" },
            { emit: "stdout", level: "warn" },
          ]
        : [{ emit: "stdout", level: "error" }],
  });

if (env.NODE_ENV === "development") {
  global.__prismaClient = prisma;
  prisma.$on("query", (event) => {
    logger.debug(
      { duration: event.duration, query: event.query, params: event.params },
      "prisma.query",
    );
  });
}

export async function transaction<T>(fn: (client: PrismaClient) => Promise<T>): Promise<T> {
  return prisma.$transaction(async (tx) => fn(tx as PrismaClient));
}

