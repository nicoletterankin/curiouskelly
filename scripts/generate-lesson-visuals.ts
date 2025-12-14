
import * as dotenv from 'dotenv';
import * as fs from 'fs';
import * as path from 'path';
import { createClient } from '@supabase/supabase-js';
import { GoogleGenerativeAI } from '@google/generative-ai';

dotenv.config();

// Configuration
const CONFIG = {
  GEMINI_API_KEY: process.env.GEMINI_API_KEY || process.env.GOOGLE_AI_API_KEY || process.env.GOOGLE_API_KEY,
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_KEY, // Prefer service role for writing
  OUTPUT_DIR: path.join(process.cwd(), 'public/generated-visuals'),
  LESSONS_FILE: path.join(process.cwd(), 'lessons/365_day_calendar.json'),
  BUCKET_NAME: 'lesson-visuals'
};

// Initialize Clients
if (!CONFIG.GEMINI_API_KEY) {
  console.error("❌ Missing GEMINI_API_KEY");
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(CONFIG.GEMINI_API_KEY);
const supabase = CONFIG.SUPABASE_URL && CONFIG.SUPABASE_KEY 
  ? createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY) 
  : null;

interface Lesson {
  day: number;
  title: string;
  objective: string;
  [key: string]: any;
}

interface VisualPlan {
  thumbnailPrompt: string;
  infographics: Array<{
    title: string;
    description: string;
    prompt: string;
    type: 'diagram' | 'chart' | 'illustration'
  }>;
}

class LessonVisualGenerator {
  
  async generateVisualPlan(lesson: Lesson): Promise<VisualPlan | null> {
    console.log(`\n🧠 Generating Visual Plan for Day ${lesson.day}: ${lesson.title}`);
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

    const prompt = `
      You are an expert educational visual designer.
      Create a visual plan for this lesson:
      Title: ${lesson.title}
      Objective: ${lesson.objective}

      I need:
      1. One "Netflix-style" thumbnail image prompt (16:9). High drama, curiosity-inducing, cinematic lighting.
      2. Two to three (2-3) supporting infographic prompts (16:9 aspect ratio suitable for slides/video overlay). These should be diagrams, charts, or conceptual illustrations that explain key concepts.

      Output JSON format:
      {
        "thumbnailPrompt": "detailed image generation prompt for Imagen 3...",
        "infographics": [
          {
            "title": "Short title for the visual",
            "description": "Explanation of what this shows",
            "prompt": "detailed image generation prompt for Imagen 3...",
            "type": "diagram" | "chart" | "illustration"
          }
        ]
      }
    `;

    try {
      const result = await model.generateContent({
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: { responseMimeType: "application/json" }
      });
      const responseText = result.response.text();
      return JSON.parse(responseText) as VisualPlan;
    } catch (error) {
      console.error(`❌ Failed to generate visual plan: ${error}`);
      return null;
    }
  }

  async generateImage(prompt: string, filename: string): Promise<string | null> {
    const outputPath = path.join(CONFIG.OUTPUT_DIR, filename);
    
    // Check if exists
    if (fs.existsSync(outputPath)) {
      console.log(`  ⏭️  Skipping existing image: ${filename}`);
      return outputPath;
    }

    console.log(`  🎨 Generating Image: ${filename}`);
    
    // Using Imagen 4 via Gemini API (REST)
    const url = `https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-generate-001:predict?key=${CONFIG.GEMINI_API_KEY}`;
    
    const requestBody = {
      instances: [{ prompt: prompt }],
      parameters: {
        sampleCount: 1,
        aspectRatio: "16:9",
        personGeneration: "allow_adult",
        safetySetting: "block_low_and_above"
      }
    };

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`  ❌ Imagen API Error: ${response.status} - ${errorText}`);
        return null;
      }

      const data = await response.json() as any;
      if (data.predictions && data.predictions[0]?.bytesBase64Encoded) {
        const buffer = Buffer.from(data.predictions[0].bytesBase64Encoded, 'base64');
        fs.mkdirSync(path.dirname(outputPath), { recursive: true });
        fs.writeFileSync(outputPath, buffer);
        console.log(`  ✅ Saved local: ${filename}`);
        return outputPath;
      }
      return null;
    } catch (error) {
      console.error(`  ❌ Image generation failed: ${error}`);
      return null;
    }
  }

  async uploadToSupabase(filePath: string, remotePath: string): Promise<string | null> {
    if (!supabase) {
      console.warn("  ⚠️ Supabase not configured. Skipping upload.");
      return null;
    }

    try {
      const fileContent = fs.readFileSync(filePath);
      const { data, error } = await supabase
        .storage
        .from(CONFIG.BUCKET_NAME)
        .upload(remotePath, fileContent, {
          contentType: 'image/png',
          upsert: true
        });

      if (error) throw error;

      const { data: publicUrlData } = supabase
        .storage
        .from(CONFIG.BUCKET_NAME)
        .getPublicUrl(remotePath);
      
      console.log(`  ☁️  Uploaded to Supabase: ${remotePath}`);
      return publicUrlData.publicUrl;
    } catch (error) {
      console.error(`  ❌ Upload failed: ${error}`);
      return null;
    }
  }

  async updateLessonRecord(day: number, visuals: any) {
    if (!supabase) return;
    
    // Update logic here depending on schema. 
    // Assuming a 'visuals' jsonb column or similar in a 'lessons' table
    try {
        /* 
        // Example update (commented out until schema confirmed)
        const { error } = await supabase
            .from('lessons')
            .update({ visuals: visuals })
            .eq('day', day);
        if (error) throw error;
        console.log(`  💾 Database updated for Day ${day}`);
        */
    } catch (error) {
        console.error(`  ❌ DB Update failed: ${error}`);
    }
  }

  async processDay(dayNumber: number) {
    // Load lesson
    const lessonsData = JSON.parse(fs.readFileSync(CONFIG.LESSONS_FILE, 'utf8'));
    const lesson = lessonsData.lessons?.find((l: any) => l.day === dayNumber);
    
    if (!lesson) {
      console.error(`Lesson Day ${dayNumber} not found.`);
      return;
    }

    // 1. Generate Visual Plan
    const plan = await this.generateVisualPlan(lesson);
    if (!plan) return;

    const results = {
      thumbnail: null as string | null,
      infographics: [] as string[]
    };

    // 2. Generate Thumbnail
    const thumbFilename = `day-${dayNumber}/thumbnail.png`;
    const localThumb = await this.generateImage(plan.thumbnailPrompt, thumbFilename);
    if (localThumb) {
      const remoteThumb = await this.uploadToSupabase(localThumb, thumbFilename);
      results.thumbnail = remoteThumb;
    }

    // 3. Generate Infographics
    for (let i = 0; i < plan.infographics.length; i++) {
      const info = plan.infographics[i];
      const filename = `day-${dayNumber}/infographic-${i + 1}.png`;
      const localImg = await this.generateImage(info.prompt, filename);
      if (localImg) {
        const remoteImg = await this.uploadToSupabase(localImg, filename);
        if (remoteImg) results.infographics.push(remoteImg);
      }
    }

    console.log(`✅ Day ${dayNumber} Completed.`);
    console.log(JSON.stringify(results, null, 2));
  }
}

// CLI
const args = process.argv.slice(2);
const generator = new LessonVisualGenerator();

if (args.includes('--help') || args.length === 0) {
    console.log("Usage: npx tsx scripts/generate-lesson-visuals.ts --day <number>");
    process.exit(0);
}

const dayArgIndex = args.indexOf('--day');
if (dayArgIndex !== -1 && args[dayArgIndex + 1]) {
    const day = parseInt(args[dayArgIndex + 1]);
    generator.processDay(day);
} else {
    console.log("Please provide a day number with --day");
}
