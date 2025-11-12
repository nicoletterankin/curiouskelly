import { z } from "zod";

export const Region = z.enum(["AMER", "EUROPE", "APAC"]);
export type Region = z.infer<typeof Region>;

export const Locale = z
  .string()
  .min(2)
  .regex(/^[a-z]{2}(?:-[A-Z]{2})?$/, "Locale must follow BCP-47 format.");

export const LessonAsset = z.object({
  type: z.enum(["video", "audio", "image", "document"]),
  url: z.string().url(),
  checksum: z.string(),
  sizeBytes: z.number().nonnegative(),
  contentType: z.string()
});

export const LessonManifest = z.object({
  id: z.string().uuid(),
  topic: z.string(),
  locale: Locale,
  title: z.string(),
  synopsis: z.string(),
  durationMinutes: z.number().int().positive(),
  ageRating: z.string(),
  assets: z.array(LessonAsset),
  checksum: z.string(),
  availableFrom: z.string().datetime(),
  availableUntil: z.string().datetime()
});
export type LessonManifest = z.infer<typeof LessonManifest>;

export const ScheduleSlot = z.object({
  id: z.string().uuid(),
  topic: z.string(),
  region: Region,
  startTime: z.string().datetime(),
  capacity: z.number().int().positive(),
  instructorId: z.string().uuid(),
  allowOverflow: z.boolean().default(false)
});
export type ScheduleSlot = z.infer<typeof ScheduleSlot>;

export const JoinResponse = z.discriminatedUnion("mode", [
  z.object({
    mode: z.literal("interactive"),
    lobbyToken: z.string(),
    policy: z.object({
      maxDevices: z.number().int().positive(),
      maxInteractivePerDay: z.number().int().nonnegative(),
      features: z.array(z.string())
    }),
    roomHint: z.string()
  }),
  z.object({
    mode: z.literal("spectator"),
    spectator: z.object({
      token: z.string(),
      playbackUrl: z.string().url(),
      expiresAt: z.string().datetime()
    }),
    policy: z.object({
      maxDevices: z.number().int().positive(),
      maxInteractivePerDay: z.number().int().nonnegative(),
      features: z.array(z.string())
    }),
    roomHint: z.string()
  })
]);
export type JoinResponse = z.infer<typeof JoinResponse>;

export const AttendanceEvent = z.object({
  sessionId: z.string().uuid(),
  userId: z.string().uuid(),
  joinedAt: z.string().datetime(),
  leftAt: z.string().datetime().optional()
});
export type AttendanceEvent = z.infer<typeof AttendanceEvent>;

export const FeedbackPayload = z.object({
  sessionId: z.string().uuid(),
  topic: z.string(),
  rating: z.number().int().min(1).max(10),
  comment: z.string().max(1500).optional(),
  submittedAt: z.string().datetime()
});
export type FeedbackPayload = z.infer<typeof FeedbackPayload>;

export const ClassroomEvent = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("presence.join"),
    userId: z.string().uuid(),
    sessionId: z.string().uuid(),
    timestamp: z.string().datetime()
  }),
  z.object({
    type: z.literal("presence.leave"),
    userId: z.string().uuid(),
    sessionId: z.string().uuid(),
    timestamp: z.string().datetime()
  }),
  z.object({
    type: z.literal("chat.message"),
    userId: z.string().uuid(),
    sessionId: z.string().uuid(),
    messageId: z.string().uuid(),
    content: z.string().max(500),
    timestamp: z.string().datetime()
  }),
  z.object({
    type: z.literal("poll.start"),
    sessionId: z.string().uuid(),
    pollId: z.string().uuid(),
    question: z.string().max(300),
    options: z.array(z.object({ id: z.string().uuid(), label: z.string().max(150) })),
    timestamp: z.string().datetime()
  }),
  z.object({
    type: z.literal("poll.vote"),
    sessionId: z.string().uuid(),
    pollId: z.string().uuid(),
    optionId: z.string().uuid(),
    userId: z.string().uuid(),
    timestamp: z.string().datetime()
  }),
  z.object({
    type: z.literal("moderation.mute"),
    sessionId: z.string().uuid(),
    targetUserId: z.string().uuid(),
    moderatorId: z.string().uuid(),
    reason: z.string().max(200),
    timestamp: z.string().datetime()
  }),
  z.object({
    type: z.literal("moderation.kick"),
    sessionId: z.string().uuid(),
    targetUserId: z.string().uuid(),
    moderatorId: z.string().uuid(),
    reason: z.string().max(200),
    timestamp: z.string().datetime()
  })
]);
export type ClassroomEvent = z.infer<typeof ClassroomEvent>;

export const ApiError = z.object({
  error: z.string(),
  code: z.string(),
  traceId: z.string().optional()
});
export type ApiError = z.infer<typeof ApiError>;

