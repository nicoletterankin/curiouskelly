import crypto from "node:crypto";
import { promises as fs } from "node:fs";
import { join } from "node:path";
import matter from "gray-matter";
import globby from "globby";
import { logger } from "@acme/logger";

const CONTENT_DIR = join(process.cwd(), "content", "lessons");
const OUTPUT_DIR = join(process.cwd(), "content", "packs");

const run = async () => {
  logger.info({ CONTENT_DIR }, "Building lesson packs");
  const files = await globby("**/*.md", { cwd: CONTENT_DIR });

  await fs.mkdir(OUTPUT_DIR, { recursive: true });

  for (const file of files) {
    const path = join(CONTENT_DIR, file);
    const raw = await fs.readFile(path, "utf8");
    const { data, content } = matter(raw);
    const manifest = {
      id: crypto.randomUUID(),
      topic: data.topic,
      locale: data.locale,
      title: data.title,
      synopsis: data.synopsis ?? "",
      durationMinutes: data.duration ?? 15,
      ageRating: data.age_rating ?? "G",
      assets: [],
      checksum: "todo",
      availableFrom: new Date().toISOString(),
      availableUntil: new Date().toISOString(),
    };
    const outDir = join(OUTPUT_DIR, data.topic, data.locale);
    await fs.mkdir(outDir, { recursive: true });
    await fs.writeFile(join(outDir, "lesson.md"), content);
    await fs.writeFile(join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2));
    logger.info({ topic: data.topic, locale: data.locale }, "Lesson pack generated");
  }
};

run()
  .then(() => logger.info("Lesson build complete"))
  .catch((error) => {
    logger.error(error);
    process.exit(1);
  });

