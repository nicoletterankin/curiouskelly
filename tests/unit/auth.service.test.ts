import { describe, it, expect, beforeEach, vi } from "vitest";
import { AuthService, AuthError } from "@acme/service-auth";

const prismaMock = {
  user: {
    upsert: vi.fn(),
    findUnique: vi.fn(),
  },
  magicLink: {
    create: vi.fn(),
    findFirst: vi.fn(),
    update: vi.fn(),
  },
  attendance: {
    count: vi.fn(),
  },
  subscription: {
    findFirst: vi.fn(),
    updateMany: vi.fn(),
  },
  device: {
    upsert: vi.fn(),
  },
  auditEvent: {
    create: vi.fn(),
  },
  authChallenge: {
    create: vi.fn(),
    findFirst: vi.fn(),
    delete: vi.fn(),
  },
  authPasskey: {
    upsert: vi.fn(),
    findUnique: vi.fn(),
    update: vi.fn(),
  },
};

const notificationsMock = {
  enqueue: vi.fn(),
};

vi.mock("@acme/database", () => ({
  prisma: prismaMock,
}));

vi.mock("@acme/service-notifications", () => ({
  notificationsService: notificationsMock,
}));

const service = new AuthService();

describe("AuthService", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    prismaMock.user.upsert.mockResolvedValue({
      id: "user-1",
      email: "user@example.com",
      displayName: "user",
    });
    prismaMock.magicLink.create.mockResolvedValue(undefined);
    prismaMock.magicLink.findFirst.mockResolvedValue({
      id: "ml-1",
      userId: "user-1",
      user: { id: "user-1", email: "user@example.com" },
    });
    prismaMock.magicLink.update.mockResolvedValue(undefined);
    prismaMock.attendance.count.mockResolvedValue(0);
    prismaMock.subscription.findFirst.mockResolvedValue({
      plan: { code: "PLUS" },
    });
    prismaMock.device.upsert.mockResolvedValue({ id: "device-1" });
    prismaMock.auditEvent.create.mockResolvedValue(undefined);
  });

  it("initiates a magic link and enqueues notification", async () => {
    const result = await service.initiateMagicLink("User@Example.com");
    expect(result.token).toBeDefined();
    expect(result.url).toContain(result.token);
    expect(notificationsMock.enqueue).toHaveBeenCalledWith(
      expect.objectContaining({
        userId: "user-1",
        channel: "EMAIL",
        template: "magic_link",
      }),
    );
  });

  it("enforces concurrency limits when verifying magic link", async () => {
    prismaMock.attendance.count.mockResolvedValueOnce(1);
    await expect(service.verifyMagicLink("token", "fingerprint")).rejects.toMatchObject({
      code: "CONCURRENCY_LIMIT",
    } satisfies Partial<AuthError>);
  });
});

