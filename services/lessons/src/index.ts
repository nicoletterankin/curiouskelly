import { S3Client, GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { startOfDay } from "date-fns";
import { prisma } from "@acme/database";
import { env } from "@acme/config";
import { logger } from "@acme/logger";
import { LessonManifest } from "@acme/types";

const s3 = new S3Client({
  region: "auto",
  endpoint: env.R2_ENDPOINT,
  credentials: {
    accessKeyId: env.R2_ACCESS_KEY_ID,
    secretAccessKey: env.R2_SECRET_ACCESS_KEY,
  },
});

export class LessonsService {
  async getTodayManifest(topic: string, locale: string): Promise<LessonManifest | null> {
    const now = new Date();
    const start = startOfDay(now);
    const manifest = await prisma.lessonManifest.findFirst({
      where: {
        topic: {
          topic,
        },
        locale,
        availableFrom: {
          lte: now,
        },
        availableUntil: {
          gte: start,
        },
      },
      orderBy: {
        availableFrom: "desc",
      },
      include: {
        topic: true,
      },
    });

    if (!manifest) {
      logger.warn({ topic, locale }, "LessonsService.getTodayManifest.notFound");
      return null;
    }

    return {
      id: manifest.id,
      topic,
      locale,
      title: manifest.topic.title,
      synopsis: manifest.topic.summary,
      durationMinutes: manifest.durationMinutes,
      ageRating: manifest.topic.ageRating,
      assets: await this.prefetchAssets(manifest.assetsBasePath),
      checksum: manifest.checksum,
      availableFrom: manifest.availableFrom.toISOString(),
      availableUntil: manifest.availableUntil.toISOString(),
    };
  }

  private async prefetchAssets(basePath: string) {
    const manifestKey = `${basePath}/manifest.json`;
    const htmlKey = `${basePath}/lesson.html`;

    const [manifestUrl, htmlUrl] = await Promise.all([
      this.signAsset(manifestKey),
      this.signAsset(htmlKey),
    ]);

    return [
      {
        type: "document",
        url: manifestUrl,
        checksum: "sha256-placeholder",
        sizeBytes: 0,
        contentType: "application/json",
      },
      {
        type: "document",
        url: htmlUrl,
        checksum: "sha256-placeholder",
        sizeBytes: 0,
        contentType: "text/html",
      },
    ];
  }

  private async signAsset(key: string): Promise<string> {
    const command = new GetObjectCommand({
      Bucket: env.R2_BUCKET,
      Key: key,
    });
    return getSignedUrl(s3, command, { expiresIn: 60 * 15 });
  }
}

export const lessonsService = new LessonsService();

