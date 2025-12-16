
import * as dotenv from 'dotenv';
import * as fs from 'fs';
import * as path from 'path';
import { GoogleGenerativeAI } from '@google/generative-ai';

dotenv.config();

// Configuration
const CONFIG = {
  GEMINI_API_KEY: process.env.GEMINI_API_KEY || process.env.GOOGLE_AI_API_KEY || process.env.GOOGLE_API_KEY,
  OUTPUT_DIR: path.join(process.cwd(), 'generated-prompts'),
  LESSONS_FILE: path.join(process.cwd(), 'lessons/365_day_calendar.json'),
  GENERATED_LESSONS_DIR: path.join(process.cwd(), 'generated/lessons'),
  START_DAY: 1,
  END_DAY: 50
};

if (!CONFIG.GEMINI_API_KEY) {
  console.error("❌ Missing GEMINI_API_KEY");
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(CONFIG.GEMINI_API_KEY);

// Helper to read JSON safely
function readJSON(filePath: string) {
  try {
    if (fs.existsSync(filePath)) {
      return JSON.parse(fs.readFileSync(filePath, 'utf8'));
    }
  } catch (e) {
    console.error(`Error reading ${filePath}:`, e);
  }
  return null;
}

interface PromptOutput {
  day: number;
  title: string;
  thumbnailPrompt: string;
  infographicPrompts: string[];
}

async function generatePromptsForDay(day: number, calendarLesson: any) {
  console.log(`\n📝 Generating prompts for Day ${day}: ${calendarLesson.title}`);

  const paddedDay = String(day).padStart(3, '0');
  const detailFile = path.join(CONFIG.GENERATED_LESSONS_DIR, `day-${paddedDay}.json`);
  const detailLesson = readJSON(detailFile);

  const context = {
    title: calendarLesson.title,
    objective: calendarLesson.objective,
    universalTruth: detailLesson?.meta?.universalTruth || detailLesson?.universalTruth || "N/A",
    keyConcepts: detailLesson?.ageVariants?.['6-12']?.phases ? 
      [
        detailLesson.ageVariants['6-12'].phases.q1.en,
        detailLesson.ageVariants['6-12'].phases.q2.en,
        detailLesson.ageVariants['6-12'].phases.q3.en
      ].join('\n') : "N/A"
  };

  const systemPrompt = `
    You are an expert visual designer for an educational platform.
    Create image generation prompts for the following lesson:
    
    Topic: ${context.title}
    Objective: ${context.objective}
    Core Message: ${context.universalTruth}
    Key Concepts:
    ${context.keyConcepts}

    I need:
    1. ONE "Netflix-style" thumbnail prompt (16:9).
       - Style: High-end documentary, cinematic lighting, photorealistic or highly detailed 3D render.
       - Vibe: Curiosity-inducing, dramatic, "Must Watch".
       - Content: Represent the topic metaphorically or directly but beautifully.

    2. THREE infographic prompts (16:9).
       - Style: Clean, modern, high-contrast, suitable for overlaying text later (but don't include text in the image).
       - Content: Visualizing the key concepts mentioned above.
       - Type: Diagrams, cross-sections, flowcharts, or conceptual 3D illustrations.

    Output pure JSON:
    {
      "thumbnailPrompt": "...",
      "infographicPrompts": ["...", "...", "..."]
    }
  `;

  const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

  try {
    const result = await model.generateContent({
      contents: [{ role: "user", parts: [{ text: systemPrompt }] }],
      generationConfig: { responseMimeType: "application/json" }
    });

    const output = JSON.parse(result.response.text());
    
    const resultData: PromptOutput = {
      day,
      title: context.title,
      thumbnailPrompt: output.thumbnailPrompt,
      infographicPrompts: output.infographicPrompts
    };

    const outputFile = path.join(CONFIG.OUTPUT_DIR, `day-${day}.json`);
    fs.mkdirSync(path.dirname(outputFile), { recursive: true });
    fs.writeFileSync(outputFile, JSON.stringify(resultData, null, 2));
    
    console.log(`  ✅ Saved to ${outputFile}`);
    
  } catch (error) {
    console.error(`  ❌ Failed for Day ${day}:`, error);
  }
}

async function main() {
  const calendarData = readJSON(CONFIG.LESSONS_FILE);
  if (!calendarData || !calendarData.lessons) {
    console.error("❌ Could not load calendar lessons");
    process.exit(1);
  }

  console.log(`🚀 Starting batch prompt generation for Days ${CONFIG.START_DAY}-${CONFIG.END_DAY}`);
  console.log(`Using model: gemini-2.0-flash`);

  for (let day = CONFIG.START_DAY; day <= CONFIG.END_DAY; day++) {
    const lesson = calendarData.lessons.find((l: any) => l.day === day);
    if (lesson) {
      await generatePromptsForDay(day, lesson);
      // Small delay to be nice to the API (though flash is fast/high-limit)
      await new Promise(r => setTimeout(r, 200)); 
    } else {
      console.warn(`  ⚠️ Lesson not found for Day ${day}`);
    }
  }
  
  console.log("\n✨ Batch generation complete!");
}

main();



