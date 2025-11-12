import { addHours, eachHourOfInterval, startOfHour } from "date-fns";
import { Queue } from "bullmq";
import Redis from "ioredis";
import { prisma } from "@acme/database";
import { env } from "@acme/config";
import { logger } from "@acme/logger";
import { Region, ScheduleSlot } from "@acme/types";

const redis = new Redis(env.REDIS_URL);

export interface NextSlotOptions {
  topic: string;
  region: Region;
  limit?: number;
}

export class ScheduleService {
  private readonly seedQueue = new Queue("schedule.seed", { connection: { url: env.REDIS_URL } });

  async getNextSlots({ topic, region, limit = 3 }: NextSlotOptions): Promise<ScheduleSlot[]> {
    const now = new Date();
    const slots = await prisma.scheduleSlot.findMany({
      where: {
        topic: { topic },
        region,
        startTime: {
          gte: now,
        },
      },
      orderBy: {
        startTime: "asc",
      },
      take: limit,
      include: {
        topic: true,
      },
    });

    return slots.map((slot) => ({
      id: slot.id,
      topic: slot.topic.topic,
      region: slot.region as Region,
      startTime: slot.startTime.toISOString(),
      capacity: slot.capacity,
      instructorId: slot.instructorId,
      allowOverflow: slot.allowOverflow,
    }));
  }

  async scheduleSeed(region: Region, topic: string) {
    await this.seedQueue.add(
      "seed",
      { region, topic },
      {
        repeat: {
          pattern: "0 * * * *", // hourly
        },
      },
    );
  }

  async seedMissingSlots(region: Region, topic: string) {
    const now = startOfHour(new Date());
    const horizon = addHours(now, 24 * 14); // two-week rolling
    const hours = eachHourOfInterval({ start: now, end: horizon });

    for (const hour of hours) {
      const existing = await prisma.scheduleSlot.findFirst({
        where: {
          topic: { topic },
          region,
          startTime: hour,
        },
      });
      if (existing) continue;

      const instructor = await prisma.instructor.findFirst({
        where: {
          isActive: true,
          skills: {
            has: topic,
          },
        },
        orderBy: {
          updatedAt: "asc",
        },
      });

      if (!instructor) {
        logger.warn({ topic, region, hour }, "ScheduleService.seedMissingSlots.noInstructor");
        continue;
      }

      await prisma.scheduleSlot.create({
        data: {
          topic: {
            connect: {
              topic,
            },
          },
          region,
          startTime: hour,
          capacity: 150,
          instructor: {
            connect: {
              id: instructor.id,
            },
          },
          allowOverflow: false,
        },
      });
      logger.info({ topic, region, hour }, "ScheduleService.seedMissingSlots.created");
    }
  }

  async reserveSeat(slotId: string, userId: string): Promise<boolean> {
    const lockKey = `schedule:${slotId}:lock`;
    const seatsKey = `schedule:${slotId}:seats`;

    const lock = await redis.set(lockKey, userId, "PX", 5000, "NX");
    if (!lock) {
      return false;
    }

    try {
      const seatsTaken = await redis.incr(seatsKey);
      const slot = await prisma.scheduleSlot.findUnique({ where: { id: slotId } });
      if (!slot) return false;

      if (seatsTaken <= slot.capacity) {
        return true;
      }

      if (slot.allowOverflow) {
        return true;
      }

      await redis.decr(seatsKey);
      return false;
    } finally {
      await redis.del(lockKey);
    }
  }
}

export const scheduleService = new ScheduleService();

