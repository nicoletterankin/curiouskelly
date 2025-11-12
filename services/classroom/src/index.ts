import { AccessToken } from "livekit-server-sdk";
import { Queue } from "bullmq";
import Redis from "ioredis";
import { addSeconds } from "date-fns";
import { prisma } from "@acme/database";
import { env } from "@acme/config";
import { logger } from "@acme/logger";
import { ClassroomEvent } from "@acme/types";

const redis = new Redis(env.REDIS_URL);

export class ClassroomService {
  private readonly moderationQueue = new Queue("classroom.moderation", {
    connection: { url: env.REDIS_URL },
  });

  async ensureSessionForSlot(slotId: string, ownerId: string) {
    const existing = await prisma.session.findFirst({
      where: {
        scheduleSlotId: slotId,
        status: {
          in: ["SCHEDULED", "LIVE"],
        },
      },
    });

    if (existing) {
      return existing;
    }

    const created = await prisma.session.create({
      data: {
        scheduleSlot: {
          connect: { id: slotId },
        },
        owner: {
          connect: { id: ownerId },
        },
        livekitRoomId: `session-${slotId}`,
        livekitUrl: env.LIVEKIT_HOST,
        status: "SCHEDULED",
      },
    });

    return created;
  }

  async mintAccessToken(sessionId: string, userId: string, identity: string, metadata: Record<string, string>) {
    const session = await prisma.session.findUnique({
      where: { id: sessionId },
      include: {
        scheduleSlot: true,
      },
    });

    if (!session) {
      throw new Error("Session not found");
    }

    const at = new AccessToken(env.LIVEKIT_API_KEY, env.LIVEKIT_API_SECRET, {
      identity,
      metadata: JSON.stringify(metadata),
    });
    at.addGrant({
      room: session.livekitRoomId,
      roomJoin: true,
      roomAdmin: false,
      canPublish: true,
      canSubscribe: true,
    });

    await prisma.attendance.upsert({
      where: {
        sessionId_userId: {
          sessionId,
          userId,
        },
      },
      update: {
        joinedAt: new Date(),
      },
      create: {
        sessionId,
        userId,
        joinedAt: new Date(),
      },
    });

    return {
      token: await at.toJwt(),
      room: session.livekitRoomId,
      url: session.livekitUrl,
    };
  }

  async recordEvent(event: ClassroomEvent) {
    const key = `session:${event.sessionId}:events`;
    await redis.rpush(key, JSON.stringify(event));
    await redis.expire(key, 60 * 60 * 6); // 6 hours

    if (event.type.startsWith("moderation.")) {
      await this.moderationQueue.add("event", event);
    }
  }

  async closeSession(sessionId: string) {
    await prisma.session.update({
      where: { id: sessionId },
      data: {
        status: "ENDED",
      },
    });
    logger.info({ sessionId }, "ClassroomService.closeSession");
  }

  async closeSessionByRoom(roomName: string) {
    await prisma.session.updateMany({
      where: { livekitRoomId: roomName },
      data: { status: "ENDED" },
    });
    logger.info({ roomName }, "ClassroomService.closeSessionByRoom");
  }

  async markParticipantLeft(sessionId: string, userId: string) {
    await prisma.attendance.update({
      where: {
        sessionId_userId: {
          sessionId,
          userId,
        },
      },
      data: {
        leftAt: new Date(),
      },
    });
  }

  async createOverflowSpectatorAccess(sessionId: string): Promise<{
    token: string;
    playbackUrl: string;
    expiresAt: string;
  }> {
    const session = await prisma.session.findUnique({
      where: { id: sessionId },
    });

    if (!session) {
      throw new Error("Session not found");
    }

    const expiresAt = addSeconds(new Date(), 60 * 30);
    const accessToken = new AccessToken(env.LIVEKIT_API_KEY, env.LIVEKIT_API_SECRET, {
      identity: `spectator-${sessionId}-${Date.now()}`,
      metadata: JSON.stringify({ spectator: true }),
    });
    accessToken.addGrant({
      room: session.livekitRoomId,
      roomJoin: true,
      roomAdmin: false,
      canPublish: false,
      canSubscribe: true,
    });

    const token = await accessToken.toJwt();
    const playbackUrl = new URL(`/hls/${session.livekitRoomId}.m3u8`, env.LIVEKIT_HOST).toString();

    await redis.hmset(`session:${sessionId}:overflow`, {
      token,
      expiresAt: expiresAt.toISOString(),
      playbackUrl,
    });
    await redis.expire(`session:${sessionId}:overflow`, 60 * 30);

    return { token, playbackUrl, expiresAt: expiresAt.toISOString() };
  }

  async getReplayUrl(sessionId: string): Promise<string | null> {
    const replay = await prisma.replay.findUnique({ where: { sessionId } });
    return replay?.recordingUrl ?? null;
  }

  async markParticipantLeftByRoom(roomName: string, userId: string) {
    const session = await prisma.session.findFirst({
      where: { livekitRoomId: roomName },
    });
    if (!session) return;

    await this.markParticipantLeft(session.id, userId);
  }
}

export const classroomService = new ClassroomService();

