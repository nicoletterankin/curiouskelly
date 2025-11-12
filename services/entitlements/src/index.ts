import Redis from "ioredis";
import { Queue, QueueEvents } from "bullmq";
import { prisma } from "@acme/database";
import { env } from "@acme/config";
import { logger } from "@acme/logger";

const CACHE_TTL_SECONDS = 60;

export interface EntitlementFeatures {
  planCode: string;
  features: Record<string, number | boolean>;
}

class EntitlementsService {
  private readonly redis: Redis;

  private readonly syncQueue: Queue;

  private readonly queueEvents: QueueEvents;

  constructor() {
    this.redis = new Redis(env.REDIS_URL);
    this.syncQueue = new Queue("entitlements.sync", {
      connection: { url: env.REDIS_URL },
    });
    this.queueEvents = new QueueEvents("entitlements.sync", { connection: { url: env.REDIS_URL } });
    this.queueEvents.on("completed", ({ jobId }) => logger.debug({ jobId }, "entitlements.sync.completed"));
    this.queueEvents.on("failed", ({ jobId, failedReason }) =>
      logger.error({ jobId, failedReason }, "entitlements.sync.failed"),
    );
  }

  async getFeaturesForUser(userId: string): Promise<EntitlementFeatures> {
    const cacheKey = `entitlements:${userId}`;
    const cached = await this.redis.get(cacheKey);
    if (cached) {
      return JSON.parse(cached) as EntitlementFeatures;
    }

    const subscription = await prisma.subscription.findFirst({
      where: {
        userId,
        status: "ACTIVE",
      },
      include: {
        plan: {
          include: {
            features: {
              include: { feature: true },
            },
          },
        },
      },
      orderBy: {
        createdAt: "desc",
      },
    });

    if (!subscription) {
      const freeFeatures: EntitlementFeatures = {
        planCode: "FREE",
        features: {
          maxInteractivePerDay: 2,
          hasReplay: false,
          maxConcurrentDevices: 1,
        },
      };
      await this.redis.set(cacheKey, JSON.stringify(freeFeatures), "EX", CACHE_TTL_SECONDS);
      return freeFeatures;
    }

    const features = subscription.plan.features.reduce<Record<string, number | boolean>>((acc, feature) => {
      acc[feature.feature.code] = feature.limit ?? true;
      return acc;
    }, {});

    const value: EntitlementFeatures = {
      planCode: subscription.plan.code,
      features,
    };

    await this.redis.set(cacheKey, JSON.stringify(value), "EX", CACHE_TTL_SECONDS);
    return value;
  }

  async scheduleSync(stripeCustomerId: string) {
    await this.syncQueue.add("stripe-sync", { stripeCustomerId }, { attempts: 5, backoff: 1000 });
  }
}

export const entitlementsService = new EntitlementsService();

