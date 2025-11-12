import { prisma } from "@acme/database";
import { logger } from "@acme/logger";

export class OpsService {
  async listUpcomingSessions(limit = 20) {
    const sessions = await prisma.session.findMany({
      where: {
        createdAt: {
          gte: new Date(),
        },
      },
      include: {
        scheduleSlot: true,
      },
      take: limit,
    });
    return sessions;
  }

  async flagInstructorNoShow(sessionId: string) {
    await prisma.auditEvent.create({
      data: {
        actorType: "system",
        action: "instructor.no_show",
        targetType: "session",
        targetId: sessionId,
      },
    });
    logger.warn({ sessionId }, "OpsService.flagInstructorNoShow");
  }
}

export const opsService = new OpsService();

