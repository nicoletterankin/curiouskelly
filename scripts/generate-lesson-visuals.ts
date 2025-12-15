import * as dotenv from 'dotenv';
import * as fs from 'fs';
import * as path from 'path';
import { createClient } from '@supabase/supabase-js';
import { GoogleGenerativeAI } from '@google/generative-ai';

// Load env with local override first
dotenv.config({ path: '.env.local' });
dotenv.config();

const CONFIG = {
  GEMINI_API_KEY: process.env.GEMINI_API_KEY || process.env.GOOGLE_AI_API_KEY || process.env.GOOGLE_API_KEY,
  // Image generation backend:
  // - gemini: uses gemini image-generation model via @google/generative-ai (most reliable with AI Studio keys)
  // - imagen: uses Imagen REST :predict (requires Imagen model availability for your key/project)
  IMAGE_BACKEND: (process.env.IMAGE_BACKEND || 'gemini').toLowerCase(),
  GEMINI_IMAGE_MODEL: process.env.GEMINI_IMAGE_MODEL || 'gemini-2.0-flash-exp-image-generation',
  IMAGEN_MODEL: process.env.IMAGEN_MODEL || 'imagen-4.0-generate-001',
  // Prefer PUBLIC_SUPABASE_URL (remote) over SUPABASE_URL (often local dev)
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL,
  SUPABASE_SERVICE_KEY:
    process.env.SUPABASE_SERVICE_ROLE_KEY ||
    process.env.SUPABASE_SERVICE_KEY ||
    process.env.SUPABASE_SERVICE_ROLE ||
    process.env.SUPABASE_KEY,
  OUTPUT_DIR: path.join(process.cwd(), 'public', 'generated-visuals'),
  LESSONS_FILE: path.join(process.cwd(), 'lessons', '365_day_calendar.json'),
  BUCKET_NAME: process.env.LESSON_VISUALS_BUCKET || 'lesson-visuals',
  DEFAULT_DELAY_MS: Number(process.env.GEMINI_DELAY_MS || 2000),
};
const supabase =
  CONFIG.SUPABASE_URL && CONFIG.SUPABASE_SERVICE_KEY
    ? createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_SERVICE_KEY)
    : null;

interface Lesson {
  day: number;
  title: string;
  objective: string;
  // optional carry-through when loaded from DB
  _core_lesson_id?: string;
  [key: string]: any;
}

interface VisualPlan {
  thumbnailPrompt: string;
  illustrationPrompt: string;
  infographics: Array<{
    title: string;
    description: string;
    prompt: string;
    type: 'diagram' | 'chart' | 'illustration';
  }>;
}

class LessonVisualGenerator {
  private genAI: GoogleGenerativeAI;
  private delayMs: number;
  private uploadEnabled: boolean;

  constructor(opts?: { delayMs?: number; uploadEnabled?: boolean }) {
    if (!CONFIG.GEMINI_API_KEY) {
      throw new Error('Missing GEMINI_API_KEY (or GOOGLE_AI_API_KEY / GOOGLE_API_KEY)');
    }
    this.genAI = new GoogleGenerativeAI(CONFIG.GEMINI_API_KEY);
    this.delayMs = opts?.delayMs ?? CONFIG.DEFAULT_DELAY_MS;
    this.uploadEnabled = opts?.uploadEnabled ?? true;
  }

  private sleep(ms: number) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private safeString(err: unknown) {
    if (err instanceof Error) return `${err.name}: ${err.message}`;
    try {
      return JSON.stringify(err);
    } catch {
      return String(err);
    }
  }

  private extractJsonObject(text: string): string | null {
    const first = text.indexOf('{');
    const last = text.lastIndexOf('}');
    if (first === -1 || last === -1 || last <= first) return null;
    return text.slice(first, last + 1);
  }

  private async fetchWithBackoff(url: string, init: RequestInit, opts?: { retries?: number }) {
    const retries = opts?.retries ?? 5;

    for (let attempt = 0; attempt <= retries; attempt++) {
      const res = await fetch(url, init);
      if (res.ok) return res;

      // Retry on rate limit / transient errors
      if ([408, 429, 500, 502, 503, 504].includes(res.status) && attempt < retries) {
        const base = 800 * Math.pow(2, attempt);
        const jitter = Math.floor(Math.random() * 250);
        const wait = Math.min(15_000, base + jitter);
        await this.sleep(wait);
        continue;
      }

      return res;
    }

    return fetch(url, init);
  }

  private async getLessonFromDatabase(dayNumber: number): Promise<Lesson | null> {
    if (!supabase) return null;

    const { data: core, error: coreError } = await supabase
      .from('core_lessons')
      .select('id, day_number, topic, universal_truth')
      .eq('day_number', dayNumber)
      .single();

    if (coreError || !core) return null;

    let objective = core.universal_truth || '';

    // lesson_shards is optional; it stores more explicit objectives sometimes
    try {
      const { data: shard } = await supabase
        .from('lesson_shards')
        .select('learning_objective')
        .eq('day', dayNumber)
        .maybeSingle();

      if (shard?.learning_objective) objective = shard.learning_objective;
    } catch {
      // ignore
    }

    return {
      day: core.day_number,
      title: core.topic,
      objective,
      _core_lesson_id: core.id,
    };
  }

  private getLessonFromFile(dayNumber: number): Lesson | null {
    if (!fs.existsSync(CONFIG.LESSONS_FILE)) return null;

    const lessonsData = JSON.parse(fs.readFileSync(CONFIG.LESSONS_FILE, 'utf8'));
    const lesson = lessonsData.lessons?.find((l: any) => l.day === dayNumber);

    if (!lesson) return null;

    return {
      day: lesson.day,
      title: lesson.title,
      objective: lesson.objective || lesson.learning_objective || '',
    };
  }

  async generateVisualPlan(lesson: Lesson): Promise<VisualPlan | null> {
    console.log(`\n🧠 Generating Visual Plan for Day ${lesson.day}: ${lesson.title}`);
    const model = this.genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });

    const prompt = `
You are an expert educational visual designer.

Create a visual plan for this lesson.
IMPORTANT constraints:
- No text overlay, no captions, no words, no logos.
- 16:9 composition for all images.
- Professional, modern, clean.

Lesson:
Title: ${lesson.title}
Objective: ${lesson.objective}

I need:
1) One Netflix-style thumbnail prompt (cinematic lighting, curiosity-inducing).
2) One topic illustration prompt (clean educational illustration used inside the lesson).
3) Two to three infographic prompts (diagrams/charts/conceptual illustrations). Must be understandable without labels.

Output JSON only in this exact format:
{
  "thumbnailPrompt": "...",
  "illustrationPrompt": "...",
  "infographics": [
    {
      "title": "...",
      "description": "...",
      "prompt": "...",
      "type": "diagram" | "chart" | "illustration"
    }
  ]
}
    `.trim();

    try {
      const result = await model.generateContent({
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: { responseMimeType: 'application/json' },
      });

      const responseText = result.response.text();
      const json = this.extractJsonObject(responseText) || responseText;
      const parsed = JSON.parse(json) as VisualPlan;

      if (!parsed?.thumbnailPrompt || !parsed?.illustrationPrompt || !Array.isArray(parsed?.infographics)) {
        throw new Error('Model returned invalid VisualPlan JSON shape');
      }

      return parsed;
    } catch (error) {
      console.error(`❌ Failed to generate visual plan: ${this.safeString(error)}`);
      return null;
    }
  }

  async generateImage(prompt: string, filename: string): Promise<string | null> {
    const outputPath = path.join(CONFIG.OUTPUT_DIR, filename);

    if (fs.existsSync(outputPath)) {
      console.log(`  ⏭️  Skipping existing image: ${filename}`);
      return outputPath;
    }

    console.log(`  🎨 Generating Image: ${filename}`);

    const saveBuffer = (buffer: Buffer) => {
      fs.mkdirSync(path.dirname(outputPath), { recursive: true });
      fs.writeFileSync(outputPath, buffer);
      console.log(`  ✅ Saved local: ${filename}`);
      return outputPath;
    };

    const generateViaGemini = async (): Promise<string | null> => {
      try {
        // Gemini image generation via generateContent + responseModalities.
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${CONFIG.GEMINI_IMAGE_MODEL}:generateContent?key=${CONFIG.GEMINI_API_KEY}`;

        const body = {
          contents: [
            {
              role: 'user',
              parts: [
                {
                  text: [
                    'Generate one image plus a short descriptive caption.',
                    'Constraints: 16:9 widescreen composition. No text, no captions, no logos, no watermarks inside the image.',
                    'Prompt:',
                    prompt,
                  ].join('\n'),
                },
              ],
            },
          ],
          generationConfig: {
            // Some image-capable Gemini models require TEXT + IMAGE modalities.
            responseModalities: ['TEXT', 'IMAGE'],
          },
        };

        const response = await this.fetchWithBackoff(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });

        if (!response.ok) {
          const errorText = await response.text();
          console.error(`  ❌ Gemini image API Error: ${response.status} - ${errorText}`);
          return null;
        }

        const data = (await response.json()) as any;
        const parts = data?.candidates?.[0]?.content?.parts || [];
        for (const part of parts) {
          const inlineData = part?.inlineData;
          if (inlineData?.mimeType?.startsWith('image/') && inlineData?.data) {
            return saveBuffer(Buffer.from(inlineData.data, 'base64'));
          }
        }

        console.error('  ❌ Gemini image API returned no inline image data');
        return null;
      } catch (error) {
        console.error(`  ❌ Gemini image generation failed: ${this.safeString(error)}`);
        return null;
      }
    };

    const generateViaImagen = async (): Promise<{ ok: boolean; path?: string; status?: number }> => {
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${CONFIG.IMAGEN_MODEL}:predict?key=${CONFIG.GEMINI_API_KEY}`;
      const requestBody = {
        instances: [{ prompt }],
        parameters: {
          sampleCount: 1,
          aspectRatio: '16:9',
          personGeneration: 'allow_adult',
          safetySetting: 'block_only_high',
        },
      };

      try {
        const response = await this.fetchWithBackoff(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          const errorText = await response.text();
          console.error(`  ❌ Imagen API Error: ${response.status} - ${errorText}`);
          return { ok: false, status: response.status };
        }

        const data = (await response.json()) as any;
        const b64 = data.predictions?.[0]?.bytesBase64Encoded;
        if (!b64) return { ok: false };

        return { ok: true, path: saveBuffer(Buffer.from(b64, 'base64')) };
      } catch (error) {
        console.error(`  ❌ Imagen generation failed: ${this.safeString(error)}`);
        return { ok: false };
      }
    };

    // Default: Gemini image generation (works with the current AI Studio key setup)
    if (CONFIG.IMAGE_BACKEND === 'imagen') {
      const res = await generateViaImagen();
      if (res.ok && res.path) return res.path;
      // If Imagen isn't available for this key/project, fall back
      if (res.status === 404) {
        console.warn('  ⚠️ Imagen model not available for this key; falling back to Gemini image generation.');
      }
      return generateViaGemini();
    }

    return generateViaGemini();
  }

  async uploadToSupabase(filePath: string, remotePath: string): Promise<string | null> {
    if (!supabase || !this.uploadEnabled) {
      console.warn('  ⚠️ Supabase upload disabled or not configured.');
      return null;
    }

    try {
      const fileContent = fs.readFileSync(filePath);
      const { error } = await supabase.storage.from(CONFIG.BUCKET_NAME).upload(remotePath, fileContent, {
        contentType: 'image/png',
        upsert: true,
      });

      if (error) throw error;

      const { data: publicUrlData } = supabase.storage.from(CONFIG.BUCKET_NAME).getPublicUrl(remotePath);

      console.log(`  ☁️  Uploaded to Supabase: ${remotePath}`);
      return publicUrlData.publicUrl;
    } catch (error) {
      console.error(`  ❌ Upload failed: ${this.safeString(error)}`);
      return null;
    }
  }

  private async upsertLessonVisuals(dayNumber: number, patch: Record<string, any>) {
    if (!supabase) return;
    try {
      await supabase.from('lesson_visuals').upsert({ day_number: dayNumber, ...patch }, { onConflict: 'day_number' });
    } catch (err) {
      console.error(`  ❌ Failed to upsert lesson_visuals for Day ${dayNumber}: ${this.safeString(err)}`);
    }
  }

  async processDay(dayNumber: number) {
    const lesson = (await this.getLessonFromDatabase(dayNumber)) || this.getLessonFromFile(dayNumber);

    if (!lesson) {
      console.error(`❌ Lesson Day ${dayNumber} not found (DB + file fallback).`);
      return;
    }

    const coreLessonId = lesson._core_lesson_id;
    const dayStr = String(dayNumber).padStart(3, '0');

    await this.upsertLessonVisuals(dayNumber, {
      core_lesson_id: coreLessonId,
      topic: lesson.title,
      status: 'generating',
      error: null,
    });

    const plan = await this.generateVisualPlan(lesson);
    if (!plan) {
      await this.upsertLessonVisuals(dayNumber, {
        core_lesson_id: coreLessonId,
        topic: lesson.title,
        status: 'failed',
        error: 'Failed to generate visual plan',
      });
      return;
    }
    
    // Pace between plan generation and image generation
    await this.sleep(this.delayMs);

    const results = {
      thumbnail: null as string | null,
      illustration: null as string | null,
      infographics: [] as string[],
    };

    // Thumbnail
    const thumbPath = `day-${dayStr}/thumbnail.png`;
    const localThumb = await this.generateImage(plan.thumbnailPrompt, thumbPath);
    if (localThumb) {
      results.thumbnail = (await this.uploadToSupabase(localThumb, thumbPath)) || null;
    }

    await this.sleep(this.delayMs);

    // Illustration
    const illustrationPath = `day-${dayStr}/illustration.png`;
    const localIllustration = await this.generateImage(plan.illustrationPrompt, illustrationPath);
    if (localIllustration) {
      results.illustration = (await this.uploadToSupabase(localIllustration, illustrationPath)) || null;
    }

    await this.sleep(this.delayMs);

    // Infographics
    for (let i = 0; i < plan.infographics.length; i++) {
      const info = plan.infographics[i];
      const p = `day-${dayStr}/infographic-${i + 1}.png`;
      const local = await this.generateImage(info.prompt, p);
      if (local) {
        const remote = await this.uploadToSupabase(local, p);
        if (remote) results.infographics.push(remote);
      }
      await this.sleep(this.delayMs);
    }

    console.log(`✅ Day ${dayNumber} Completed.`);
    console.log(JSON.stringify(results, null, 2));

    const status = results.thumbnail || results.illustration || results.infographics.length > 0 ? 'completed' : 'failed';

    await this.upsertLessonVisuals(dayNumber, {
      core_lesson_id: coreLessonId,
      topic: lesson.title,
      thumbnail_url: results.thumbnail,
      thumbnail_path: thumbPath,
      illustration_url: results.illustration,
      illustration_path: illustrationPath,
      infographic_url: results.infographics[0] || null,
      infographic_urls: results.infographics,
      status,
      error: status === 'failed' ? 'No assets generated' : null,
    });
  }
}

async function main() {
  const args = process.argv.slice(2);

  const help = args.includes('--help') || args.includes('-h');
  const noUpload = args.includes('--no-upload') || args.includes('--skip-upload');
  const dryRun = args.includes('--dry-run');
  const delayArg = args.find((a) => a.startsWith('--delay-ms='));
  const delayMs = delayArg ? Number(delayArg.split('=')[1]) : CONFIG.DEFAULT_DELAY_MS;

  if (help || args.length === 0) {
    console.log(`
🎨 GEMINI LESSON VISUAL PIPELINE

Usage:
  npx tsx scripts/generate-lesson-visuals.ts 1 70
  npx tsx scripts/generate-lesson-visuals.ts --day 1

Options:
  --no-upload           Generate locally only (skip Supabase Storage upload)
  --delay-ms=2000       Delay between image calls (default: ${CONFIG.DEFAULT_DELAY_MS})
  --dry-run             Validate env + show settings (no API calls)

Env:
  GEMINI_API_KEY (or GOOGLE_AI_API_KEY / GOOGLE_API_KEY)
  SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL / PUBLIC_SUPABASE_URL)  [optional but required for DB+upload]
  SUPABASE_SERVICE_KEY (or SUPABASE_SERVICE_ROLE_KEY)               [optional but required for DB+upload]
  IMAGE_BACKEND (default: ${CONFIG.IMAGE_BACKEND})                  # gemini | imagen
  GEMINI_IMAGE_MODEL (default: ${CONFIG.GEMINI_IMAGE_MODEL})
  IMAGEN_MODEL (default: ${CONFIG.IMAGEN_MODEL})                    # only used if IMAGE_BACKEND=imagen
  LESSON_VISUALS_BUCKET (default: ${CONFIG.BUCKET_NAME})
`);
    return;
  }

  let startDay: number | null = null;
  let endDay: number | null = null;

  const dayFlagIndex = args.indexOf('--day');
  if (dayFlagIndex !== -1 && args[dayFlagIndex + 1]) {
    startDay = Number(args[dayFlagIndex + 1]);
    endDay = startDay;
  } else if (args.length >= 1 && /^\d+$/.test(args[0])) {
    startDay = Number(args[0]);
    endDay = args[1] && /^\d+$/.test(args[1]) ? Number(args[1]) : startDay;
  }

  if (!startDay || !endDay) {
    console.error('❌ Could not parse day/range. Run with --help for examples.');
    process.exit(1);
  }

  if (dryRun) {
    console.log('🔍 DRY RUN MODE');
    console.log(`Days: ${startDay} - ${endDay}`);
    console.log(`Gemini key: ${CONFIG.GEMINI_API_KEY ? 'present' : 'missing'}`);
    console.log(`Upload: ${noUpload ? 'disabled' : supabase ? 'enabled' : 'not configured'}`);
    console.log(`Image backend: ${CONFIG.IMAGE_BACKEND}`);
    console.log(`Gemini image model: ${CONFIG.GEMINI_IMAGE_MODEL}`);
    console.log(`Imagen model: ${CONFIG.IMAGEN_MODEL}`);
    console.log(`Delay: ${delayMs}ms`);
    return;
  }

  if (!CONFIG.GEMINI_API_KEY) {
    console.error('❌ Missing GEMINI_API_KEY (or GOOGLE_AI_API_KEY / GOOGLE_API_KEY)');
    process.exit(1);
  }

  const generator = new LessonVisualGenerator({ delayMs, uploadEnabled: !noUpload });

  console.log('🎨 GEMINI VISUAL PIPELINE');
  console.log(`Days: ${startDay} - ${endDay}`);
  console.log(`Upload: ${noUpload ? 'disabled' : supabase ? 'enabled' : 'not configured'}`);
  console.log(`Image backend: ${CONFIG.IMAGE_BACKEND}`);
  console.log(`Gemini image model: ${CONFIG.GEMINI_IMAGE_MODEL}`);
  console.log(`Imagen model: ${CONFIG.IMAGEN_MODEL}`);
  console.log(`Delay: ${delayMs}ms\n`);

  for (let day = startDay; day <= endDay; day++) {
    try {
      await generator.processDay(day);
    } catch (err) {
      console.error(`❌ Day ${day} failed: ${err instanceof Error ? err.message : String(err)}`);
    }

    await new Promise((r) => setTimeout(r, delayMs));
  }

  console.log('\n✅ Visual generation run complete');
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
