import { randomUUID } from "node:crypto";
import argon2 from "argon2";
import { SignJWT, jwtVerify, JWTPayload } from "jose";
import { addMinutes, addSeconds } from "date-fns";
import {
  generateRegistrationOptions,
  verifyRegistrationResponse,
  generateAuthenticationOptions,
  verifyAuthenticationResponse,
} from "@simplewebauthn/server";
import type {
  RegistrationResponseJSON,
  AuthenticationResponseJSON,
} from "@simplewebauthn/typescript-types";
import { prisma } from "@acme/database";
import { env } from "@acme/config";
import { logger } from "@acme/logger";
import { notificationsService } from "@acme/service-notifications";

const MAGIC_LINK_TTL_MINUTES = 15;
const SESSION_TTL_HOURS = 12;
const PASSKEY_CHALLENGE_TTL_SECONDS = 120;
const RP_ID = new URL(env.APP_ORIGIN).hostname;
const RP_NAME = "Daily Lesson Platform";

export interface SessionTokenPayload extends JWTPayload {
  sub: string;
  deviceId: string;
  planCode: string;
}

export class AuthError extends Error {
  constructor(public readonly code: string, message?: string) {
    super(message ?? code);
  }
}

export class AuthService {
  async initiateMagicLink(email: string): Promise<{ url: string; token: string; expiresAt: Date }> {
    const normalizedEmail = email.trim().toLowerCase();
    const user = await prisma.user.upsert({
      where: { email: normalizedEmail },
      update: {},
      create: {
        email: normalizedEmail,
        displayName: normalizedEmail.split("@")[0] ?? "Learner",
        region: "AMER",
      },
    });

    const token = randomUUID();
    const expiresAt = addMinutes(new Date(), MAGIC_LINK_TTL_MINUTES);

    await prisma.magicLink.create({
      data: {
        token,
        expiresAt,
        userId: user.id,
      },
    });

    const link = new URL(env.MAGIC_LINK_BASE_URL);
    link.searchParams.set("token", token);

    await notificationsService.enqueue({
      userId: user.id,
      channel: "EMAIL",
      template: "magic_link",
      payload: {
        url: link.toString(),
        expiresAt: expiresAt.toISOString(),
        email: normalizedEmail,
      },
    });

    logger.info({ userId: user.id }, "AuthService.magicLinkCreated");
    return { url: link.toString(), token, expiresAt };
  }

  async verifyMagicLink(token: string, deviceFingerprint: string): Promise<string> {
    const magicLink = await prisma.magicLink.findFirst({
      where: {
        token,
        consumed: false,
        expiresAt: {
          gt: new Date(),
        },
      },
      include: {
        user: true,
      },
    });

    if (!magicLink) {
      throw new AuthError("MAGIC_LINK_INVALID", "Magic link is invalid or expired");
    }

    await this.ensureInteractiveConcurrency(magicLink.userId);

    await prisma.magicLink.update({
      where: { id: magicLink.id },
      data: { consumed: true },
    });

    const deviceId = await this.bindDevice(magicLink.userId, deviceFingerprint);
    const planCode = await this.resolvePlanCode(magicLink.userId);

    const tokenPayload: SessionTokenPayload = {
      sub: magicLink.userId,
      deviceId,
      planCode,
      exp: Math.floor(addMinutes(new Date(), SESSION_TTL_HOURS * 60).getTime() / 1000),
    };

    return this.issueSessionToken(tokenPayload);
  }

  async generatePasskeyRegistrationOptions(userId: string) {
    const user = await prisma.user.findUnique({
      where: { id: userId },
      include: { passkeys: true },
    });
    if (!user) {
      throw new AuthError("USER_NOT_FOUND", "User not found");
    }

    const options = await generateRegistrationOptions({
      rpName: RP_NAME,
      rpID: RP_ID,
      userID: user.id,
      userName: user.email,
      userDisplayName: user.displayName,
      attestationType: "none",
      timeout: 60000,
      excludeCredentials: user.passkeys.map((credential) => ({
        id: Buffer.from(credential.credentialId, "base64url"),
        type: "public-key",
      })),
    });

    await this.storeChallenge({
      userId,
      challenge: options.challenge,
      type: "REGISTRATION",
    });

    return options;
  }

  async completePasskeyRegistration(userId: string, response: RegistrationResponseJSON) {
    const challenge = await this.consumeChallenge({
      userId,
      type: "REGISTRATION",
    });

    if (!challenge) {
      throw new AuthError("CHALLENGE_NOT_FOUND", "Registration challenge missing or expired");
    }

    const verification = await verifyRegistrationResponse({
      response,
      expectedChallenge: challenge.challenge,
      expectedOrigin: env.APP_ORIGIN,
      expectedRPID: RP_ID,
    });

    if (!verification.verified || !verification.registrationInfo) {
      throw new AuthError("PASSKEY_REGISTRATION_FAILED", "Failed to verify passkey registration");
    }

    const {
      credentialID,
      credentialPublicKey,
      counter,
      credentialDeviceType,
      credentialBackedUp,
    } = verification.registrationInfo;

    await prisma.authPasskey.upsert({
      where: {
        credentialId: Buffer.from(credentialID).toString("base64url"),
      },
      create: {
        userId,
        credentialId: Buffer.from(credentialID).toString("base64url"),
        publicKey: credentialPublicKey,
        counter,
        deviceBinding: credentialDeviceType,
      },
      update: {
        counter,
        deviceBinding: credentialDeviceType,
        userId,
      },
    });

    logger.info({ userId }, "AuthService.passkeyRegistered");
    return verification;
  }

  async generatePasskeyAuthenticationOptions(email: string) {
    const user = await prisma.user.findUnique({
      where: { email },
      include: { passkeys: true },
    });

    if (!user || user.passkeys.length === 0) {
      throw new AuthError("PASSKEY_NOT_FOUND", "No passkeys registered for user");
    }

    const options = await generateAuthenticationOptions({
      rpID: RP_ID,
      timeout: 60000,
      allowCredentials: user.passkeys.map((credential) => ({
        id: Buffer.from(credential.credentialId, "base64url"),
        type: "public-key",
      })),
    });

    await this.storeChallenge({
      userId: user.id,
      challenge: options.challenge,
      type: "AUTHENTICATION",
    });

    return { options, userId: user.id };
  }

  async completePasskeyAuthentication(
    response: AuthenticationResponseJSON,
    deviceFingerprint: string,
  ): Promise<string> {
    const credential = await prisma.authPasskey.findUnique({
      where: {
        credentialId: Buffer.from(response.id, "base64url").toString("base64url"),
      },
    });

    if (!credential) {
      throw new AuthError("PASSKEY_NOT_FOUND", "Unknown credential");
    }

    const challenge = await this.consumeChallenge({
      userId: credential.userId,
      type: "AUTHENTICATION",
    });

    if (!challenge) {
      throw new AuthError("CHALLENGE_NOT_FOUND", "Authentication challenge missing");
    }

    const verification = await verifyAuthenticationResponse({
      response,
      expectedChallenge: challenge.challenge,
      expectedOrigin: env.APP_ORIGIN,
      expectedRPID: RP_ID,
      authenticator: {
        credentialID: Buffer.from(credential.credentialId, "base64url"),
        credentialPublicKey: credential.publicKey,
        counter: credential.counter,
        transports: response.response.transports,
      },
    });

    if (!verification.verified || !verification.authenticationInfo) {
      throw new AuthError("PASSKEY_AUTH_FAILED", "Failed to verify passkey login");
    }

    await prisma.authPasskey.update({
      where: { id: credential.id },
      data: {
        counter: verification.authenticationInfo.newCounter,
      },
    });

    await this.ensureInteractiveConcurrency(credential.userId);
    const deviceId = await this.bindDevice(credential.userId, deviceFingerprint);
    const planCode = await this.resolvePlanCode(credential.userId);

    const payload: SessionTokenPayload = {
      sub: credential.userId,
      deviceId,
      planCode,
      exp: Math.floor(addMinutes(new Date(), SESSION_TTL_HOURS * 60).getTime() / 1000),
    };

    return this.issueSessionToken(payload);
  }

  async issueSessionToken(payload: SessionTokenPayload): Promise<string> {
    const jwt = await new SignJWT(payload)
      .setProtectedHeader({ alg: "HS256" })
      .setIssuedAt()
      .setExpirationTime(`${SESSION_TTL_HOURS}h`)
      .sign(new TextEncoder().encode(env.JWT_SECRET));

    return jwt;
  }

  async verifySessionToken(token: string): Promise<SessionTokenPayload | null> {
    try {
      const { payload } = await jwtVerify<SessionTokenPayload>(
        token,
        new TextEncoder().encode(env.JWT_SECRET),
        { algorithms: ["HS256"] },
      );
      return payload;
    } catch (error) {
      logger.warn({ error }, "AuthService.verifySessionToken.failed");
      return null;
    }
  }

  async bindDevice(userId: string, deviceFingerprint: string): Promise<string> {
    const hashed = await argon2.hash(deviceFingerprint);
    const device = await prisma.device.upsert({
      where: { deviceFingerprint },
      update: {
        lastSeenAt: new Date(),
      },
      create: {
        userId,
        deviceFingerprint,
        platform: "unknown",
      },
    });

    await prisma.auditEvent.create({
      data: {
        actorId: userId,
        actorType: "user",
        action: "device.bound",
        targetType: "device",
        targetId: device.id,
        metadata: { hashedFingerprint: hashed },
      },
    });

    return device.id;
  }

  private async resolvePlanCode(userId: string): Promise<string> {
    const subscription = await prisma.subscription.findFirst({
      where: {
        userId,
        status: "ACTIVE",
      },
      include: {
        plan: true,
      },
      orderBy: {
        createdAt: "desc",
      },
    });

    return subscription?.plan.code ?? "FREE";
  }

  private async ensureInteractiveConcurrency(userId: string) {
    const activeSessions = await prisma.attendance.count({
      where: {
        userId,
        leftAt: null,
        session: {
          status: {
            in: ["LIVE", "SCHEDULED"],
          },
        },
      },
    });

    if (activeSessions > 0) {
      throw new AuthError("CONCURRENCY_LIMIT", "Active interactive session already in progress");
    }
  }

  private async storeChallenge({
    userId,
    challenge,
    type,
  }: {
    userId: string;
    challenge: string;
    type: "REGISTRATION" | "AUTHENTICATION";
  }) {
    await prisma.authChallenge.create({
      data: {
        userId,
        challenge,
        type,
        expiresAt: addSeconds(new Date(), PASSKEY_CHALLENGE_TTL_SECONDS),
      },
    });
  }

  private async consumeChallenge({
    userId,
    type,
    challenge,
  }: {
    userId: string;
    type: "REGISTRATION" | "AUTHENTICATION";
    challenge?: string;
  }) {
    const record = await prisma.authChallenge.findFirst({
      where: {
        userId,
        type,
        expiresAt: {
          gt: new Date(),
        },
        ...(challenge ? { challenge } : {}),
      },
      orderBy: {
        createdAt: "desc",
      },
    });

    if (!record) {
      return null;
    }

    await prisma.authChallenge.delete({
      where: { id: record.id },
    });

    return record;
  }
}

export const authService = new AuthService();
