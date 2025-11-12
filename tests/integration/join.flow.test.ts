import { describe, it, expect, beforeEach, vi } from "vitest";
import request from "supertest";
import { buildServer } from "../../apps/gateway/src/app";

const prismaMock = {
  attendance: {
    count: vi.fn(),
  },
  scheduleSlot: {
    findUnique: vi.fn(),
  },
};

const entitlementsMock = {
  getFeaturesForUser: vi.fn(),
};

const scheduleMock = {
  reserveSeat: vi.fn(),
};

const classroomMock = {
  ensureSessionForSlot: vi.fn(),
  createOverflowSpectatorAccess: vi.fn(),
  mintAccessToken: vi.fn(),
  markParticipantLeft: vi.fn(),
  markParticipantLeftByRoom: vi.fn(),
  closeSessionByRoom: vi.fn(),
};

const telemetryMock = {
  ingestServerEvent: vi.fn(),
};

vi.mock("@acme/database", () => ({
  prisma: prismaMock,
}));

vi.mock("@acme/service-entitlements", () => ({
  entitlementsService: entitlementsMock,
}));

vi.mock("@acme/service-schedule", () => ({
  scheduleService: scheduleMock,
}));

vi.mock("@acme/service-classroom", () => ({
  classroomService: classroomMock,
}));

vi.mock("@acme/service-telemetry", () => ({
  telemetryService: telemetryMock,
}));

describe("POST /v1/join", () => {
  const server = buildServer();

  beforeEach(async () => {
    vi.clearAllMocks();
    await server.ready();
    prismaMock.attendance.count.mockResolvedValue(0);
    prismaMock.scheduleSlot.findUnique.mockResolvedValue({
      id: "slot-1",
      allowOverflow: true,
    });
    entitlementsMock.getFeaturesForUser.mockResolvedValue({
      planCode: "PLUS",
      features: {
        maxInteractivePerDay: 2,
        maxConcurrentDevices: 1,
        hasReplay: true,
      },
    });
    scheduleMock.reserveSeat.mockResolvedValue(false);
    classroomMock.ensureSessionForSlot.mockResolvedValue({
      id: "session-1",
      livekitRoomId: "room-1",
    });
    classroomMock.createOverflowSpectatorAccess.mockResolvedValue({
      token: "spectator-token",
      playbackUrl: "https://livekit.example.com/hls/room-1.m3u8",
      expiresAt: new Date().toISOString(),
    });
  });

  it("returns spectator payload when overflow occurs", async () => {
    const token = server.jwt.sign({
      sub: "user-1",
      deviceId: "device-1",
      planCode: "PLUS",
    });

    const response = await request(server.server)
      .post("/v1/join")
      .set("Authorization", `Bearer ${token}`)
      .send({ slotId: "slot-1" })
      .expect(200);

    expect(response.body.mode).toBe("spectator");
    expect(response.body.spectator).toMatchObject({
      token: "spectator-token",
    });
    expect(classroomMock.createOverflowSpectatorAccess).toHaveBeenCalledWith("session-1");
    expect(telemetryMock.ingestServerEvent).toHaveBeenCalledWith(
      "session_join",
      expect.objectContaining({ overflow: true }),
    );
  });
});

