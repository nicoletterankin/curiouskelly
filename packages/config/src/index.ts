import { config as loadEnv } from "dotenv";
import { z } from "zod";

const rawEnv = loadEnv();

if (rawEnv.error) {
  // eslint-disable-next-line no-console
  console.warn("No .env file found or failed to load; relying on process.env values.");
}

const envSchema = z
  .object({
    NODE_ENV: z.enum(["development", "test", "production"]).default("development"),
    LOG_LEVEL: z.enum(["fatal", "error", "warn", "info", "debug", "trace"]).default("info"),
    DATABASE_URL: z.string().url().default("postgresql://postgres:postgres@localhost:5432/app"),
    REDIS_URL: z.string().url().default("redis://localhost:6379"),
    STRIPE_WEBHOOK_SECRET: z.string().min(1).default("whsec_dev"),
    STRIPE_SECRET_KEY: z.string().min(1).default("sk_test"),
    LIVEKIT_API_KEY: z.string().min(1).default("dev"),
    LIVEKIT_API_SECRET: z.string().min(1).default("devsecret"),
    LIVEKIT_HOST: z.string().url().default("https://livekit.example.com"),
    LIVEKIT_WEBHOOK_SECRET: z.string().min(1).default("livekit_webhook_dev"),
    R2_ENDPOINT: z.string().url().default("https://example.r2.cloudflarestorage.com"),
    R2_ACCESS_KEY_ID: z.string().min(1).default("local"),
    R2_SECRET_ACCESS_KEY: z.string().min(1).default("local"),
    R2_BUCKET: z.string().min(1).default("lesson-packs"),
    JWT_SECRET: z.string().min(32).default("dev_jwt_secret_dev_jwt_secret_dev_jwt"),
    REGION: z.enum(["AMER", "EUROPE", "APAC"]).default("AMER"),
    DEFAULT_LOCALE: z.string().default("en-US"),
    RUM_ENABLED: z
      .string()
      .optional()
      .transform((value) => value === "true")
      .pipe(z.boolean().default(false)),
    CLICKHOUSE_URL: z.string().url().default("http://localhost:8123"),
    MEILISEARCH_HOST: z.string().url().default("http://localhost:7700"),
    MEILISEARCH_API_KEY: z.string().min(1).default("masterKey"),
    CLOUDFARE_TURNSTILE_SECRET: z.string().min(1).default("turnstile_dev"),
    STAFF_SSO_ISSUER: z.string().url().default("https://sso.example.com"),
    STAFF_SSO_AUDIENCE: z.string().url().default("https://ops.example.com"),
    MAGIC_LINK_BASE_URL: z.string().url().default("https://app.example.com/magic-link"),
    APP_ORIGIN: z.string().url().default("https://app.example.com"),
    EMAIL_FROM: z.string().email().default("no-reply@example.com")
  })
  .transform((values) => ({
    ...values,
    isProduction: values.NODE_ENV === "production"
  }));

const validatedEnv = envSchema.parse(process.env);

export type AppEnv = typeof validatedEnv;

export const env: AppEnv = validatedEnv;

