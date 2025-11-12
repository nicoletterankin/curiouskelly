import { Queue } from "bullmq";
import Redis from "ioredis";
import { isWithinInterval, set } from "date-fns";
import { prisma } from "@acme/database";
import { env } from "@acme/config";
import { logger } from "@acme/logger";

export interface NotificationRequest {
  userId: string;
  template: string;
  channel: "EMAIL" | "PUSH";
  payload: Record<string, unknown>;
}

const redis = new Redis(env.REDIS_URL);

export class NotificationsService {
  private readonly queue = new Queue("notifications.dispatch", { connection: { url: env.REDIS_URL } });

  async enqueue(request: NotificationRequest) {
    await this.queue.add("dispatch", request, { attempts: 3, backoff: { type: "exponential", delay: 30_000 } });
  }

  async dispatch(request: NotificationRequest) {
    const user = await prisma.user.findUnique({ where: { id: request.userId } });
    if (!user) {
      logger.warn({ userId: request.userId }, "NotificationsService.dispatch.userMissing");
      return;
    }

    const quietHours = await this.getQuietHours(user.locale);
    if (this.isWithinQuietHours(quietHours.start, quietHours.end)) {
      await redis.zadd("notifications:queue", Date.now() + 60 * 60 * 1000, JSON.stringify(request));
      return;
    }

    await prisma.notification.create({
      data: {
        userId: request.userId,
        channel: request.channel,
        template: request.template,
        payload: request.payload,
        scheduledAt: new Date(),
        status: "SENT",
        deliveredAt: new Date(),
      },
    });
    logger.info({ request }, "NotificationsService.dispatch.sent");
  }

  private async getQuietHours(locale: string): Promise<{ start: number; end: number }> {
    const cacheKey = `quiet-hours:${locale}`;
    const cached = await redis.get(cacheKey);
    if (cached) {
      return JSON.parse(cached) as { start: number; end: number };
    }
    const defaults = { start: 22, end: 8 };
    await redis.set(cacheKey, JSON.stringify(defaults), "EX", 60 * 60);
    return defaults;
  }

  private isWithinQuietHours(startHour: number, endHour: number) {
    const now = new Date();
    const startTime = set(now, { hours: startHour, minutes: 0, seconds: 0, milliseconds: 0 });
    const endTime = set(now, { hours: endHour, minutes: 0, seconds: 0, milliseconds: 0 });

    if (startHour < endHour) {
      return isWithinInterval(now, { start: startTime, end: endTime });
    }

    return now >= startTime || now <= endTime;
  }
}

export const notificationsService = new NotificationsService();

