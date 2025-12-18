#!/usr/bin/env npx tsx
/**
 * SPECIAL LESSONS GENERATOR
 * 
 * Generates watch pages and emails for special lessons (S-001 to S-020)
 * 
 * Usage:
 *   npx tsx scripts/generate-special-lessons.ts
 *   npx tsx scripts/generate-special-lessons.ts --id S-001
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';

interface SpecialLessonData {
  meta: { 
    id: string; 
    type: string;
    topic: string; 
    emoji?: string; 
    category?: string;
  };
  headline?: string;
  universal_truth?: string;
  fun_facts?: string[];
  phases: {
    hook?: { script: string };
    fact1?: { script: string };
    fact2?: { script: string };
    wisdom?: { script: string };
    [key: string]: any;
  };
  growTrack?: {
    title: string;
    emoji?: string;
    learning_objective?: string;
    activity: string;
  };
}

function loadSpecialLesson(filename: string): SpecialLessonData | null {
  const lessonPath = path.join(process.cwd(), 'public', 'lessons', 'special', filename);
  if (!fs.existsSync(lessonPath)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(lessonPath, 'utf-8'));
}

function generateWatchPage(lesson: SpecialLessonData, filename: string): string {
  const { id, topic, emoji, category } = lesson.meta;
  const headline = lesson.headline || lesson.universal_truth || '';
  const idNum = id.replace('S-', '');
  
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${topic} — Curious Kelly</title>
  <meta name="description" content="${headline}. Special lesson from Curious Kelly.">
  
  <!-- Open Graph -->
  <meta property="og:title" content="${topic}">
  <meta property="og:description" content="${headline}">
  <meta property="og:type" content="video.other">
  <meta property="og:video" content="https://videos.curiouskelly.com/special/${id.toLowerCase()}.mp4">
  <meta property="og:image" content="https://videos.curiouskelly.com/special/${id.toLowerCase()}-thumb.jpg">
  
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
      min-height: 100vh;
      color: white;
    }
    .container {
      max-width: 900px;
      margin: 0 auto;
      padding: 40px 20px;
    }
    header {
      text-align: center;
      margin-bottom: 30px;
    }
    .special-badge {
      display: inline-block;
      background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%);
      color: white;
      padding: 6px 16px;
      border-radius: 20px;
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 15px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    h1 {
      font-size: 2rem;
      margin-bottom: 10px;
    }
    .emoji { font-size: 2.5rem; margin-bottom: 10px; display: block; }
    .headline {
      font-size: 1.1rem;
      opacity: 0.9;
      max-width: 600px;
      margin: 0 auto;
      line-height: 1.5;
    }
    .video-container {
      position: relative;
      width: 100%;
      max-width: 800px;
      margin: 30px auto;
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 20px 60px rgba(0,0,0,0.4);
      background: #000;
    }
    video {
      width: 100%;
      display: block;
    }
    .content-section {
      background: rgba(255,255,255,0.05);
      border-radius: 16px;
      padding: 30px;
      margin: 30px auto;
      max-width: 800px;
    }
    .content-section h2 {
      font-size: 1.3rem;
      margin-bottom: 20px;
      color: #e94560;
    }
    .fun-facts {
      list-style: none;
    }
    .fun-facts li {
      padding: 12px 0;
      border-bottom: 1px solid rgba(255,255,255,0.1);
      display: flex;
      align-items: flex-start;
      gap: 12px;
    }
    .fun-facts li:last-child { border-bottom: none; }
    .fun-facts li::before {
      content: "💡";
      flex-shrink: 0;
    }
    .grow-track {
      background: linear-gradient(135deg, rgba(233,69,96,0.2) 0%, rgba(255,107,107,0.1) 100%);
      border: 1px solid rgba(233,69,96,0.3);
    }
    .grow-track h2 { color: #ff6b6b; }
    .activity-text {
      line-height: 1.7;
      font-size: 1.05rem;
    }
    .cta-section {
      text-align: center;
      margin-top: 40px;
    }
    .cta-button {
      display: inline-block;
      background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%);
      color: white;
      padding: 16px 40px;
      border-radius: 30px;
      text-decoration: none;
      font-weight: 600;
      font-size: 1.1rem;
      transition: transform 0.2s, box-shadow 0.2s;
    }
    .cta-button:hover {
      transform: translateY(-2px);
      box-shadow: 0 10px 30px rgba(233,69,96,0.4);
    }
    footer {
      text-align: center;
      margin-top: 60px;
      padding: 20px;
      opacity: 0.7;
      font-size: 0.9rem;
    }
    footer a { color: #e94560; text-decoration: none; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <span class="special-badge">${category || 'Special Lesson'}</span>
      <span class="emoji">${emoji || '✨'}</span>
      <h1>${topic}</h1>
      <p class="headline">${headline}</p>
    </header>
    
    <div class="video-container">
      <video controls poster="https://videos.curiouskelly.com/special/${id.toLowerCase()}-thumb.jpg">
        <source src="https://videos.curiouskelly.com/special/${id.toLowerCase()}.mp4" type="video/mp4">
        Your browser does not support video playback.
      </video>
    </div>
    
    ${lesson.fun_facts && lesson.fun_facts.length > 0 ? `
    <section class="content-section">
      <h2>Key Insights</h2>
      <ul class="fun-facts">
        ${lesson.fun_facts.map(fact => `<li>${fact}</li>`).join('\n        ')}
      </ul>
    </section>
    ` : ''}
    
    ${lesson.growTrack ? `
    <section class="content-section grow-track">
      <h2>${lesson.growTrack.emoji || '🌱'} ${lesson.growTrack.title}</h2>
      <p class="activity-text">${lesson.growTrack.activity}</p>
    </section>
    ` : ''}
    
    <div class="cta-section">
      <a href="/calendar" class="cta-button">Back to Calendar</a>
    </div>
    
    <footer>
      <p>✨ <a href="/">Curious Kelly</a> — Daily learning for lifelong curiosity</p>
      <p style="margin-top: 10px;">© ${new Date().getFullYear()} Lesson of the Day PBC</p>
    </footer>
  </div>
</body>
</html>`;
}

function generateEmail(lesson: SpecialLessonData): string {
  const { id, topic, emoji, category } = lesson.meta;
  const headline = lesson.headline || '';
  const wisdom = lesson.phases?.wisdom?.script || lesson.universal_truth || '';
  
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${topic} — Curious Kelly</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5;">
  <table width="100%" cellpadding="0" cellspacing="0" style="max-width: 600px; margin: 0 auto; background: white;">
    <tr>
      <td style="padding: 40px 30px; text-align: center; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);">
        <p style="font-size: 14px; color: rgba(255,255,255,0.7); margin: 0 0 10px 0; text-transform: uppercase; letter-spacing: 1px;">${category || 'Special Lesson'}</p>
        <p style="font-size: 48px; margin: 0;">${emoji || '✨'}</p>
        <h1 style="color: white; font-size: 28px; margin: 15px 0 10px 0;">${topic}</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 16px; margin: 0; line-height: 1.5;">${headline}</p>
      </td>
    </tr>
    <tr>
      <td style="padding: 30px; text-align: center;">
        <a href="https://curiouskelly.com/watch/special/${id.toLowerCase()}" style="display: inline-block;">
          <img src="https://videos.curiouskelly.com/special/${id.toLowerCase()}-thumb.jpg" alt="Watch ${topic}" style="width: 100%; max-width: 500px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.15);">
        </a>
        <p style="margin: 20px 0 0 0;">
          <a href="https://curiouskelly.com/watch/special/${id.toLowerCase()}" style="display: inline-block; background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%); color: white; padding: 14px 35px; border-radius: 25px; text-decoration: none; font-weight: 600;">Watch Now</a>
        </p>
      </td>
    </tr>
    ${lesson.fun_facts && lesson.fun_facts.length > 0 ? `
    <tr>
      <td style="padding: 0 30px 30px 30px;">
        <div style="background: #f8f9fa; border-radius: 12px; padding: 25px;">
          <h2 style="margin: 0 0 15px 0; font-size: 18px; color: #333;">💡 Key Insights</h2>
          ${lesson.fun_facts.map(fact => `<p style="margin: 10px 0; color: #555; line-height: 1.5;">• ${fact}</p>`).join('')}
        </div>
      </td>
    </tr>
    ` : ''}
    ${lesson.growTrack ? `
    <tr>
      <td style="padding: 0 30px 30px 30px;">
        <div style="background: linear-gradient(135deg, rgba(233,69,96,0.1) 0%, rgba(255,107,107,0.05) 100%); border: 1px solid rgba(233,69,96,0.2); border-radius: 12px; padding: 25px;">
          <h2 style="margin: 0 0 15px 0; font-size: 18px; color: #e94560;">${lesson.growTrack.emoji || '🌱'} ${lesson.growTrack.title}</h2>
          <p style="margin: 0; color: #555; line-height: 1.6;">${lesson.growTrack.activity}</p>
        </div>
      </td>
    </tr>
    ` : ''}
    <tr>
      <td style="padding: 30px; background: #1a1a2e; text-align: center;">
        <p style="color: rgba(255,255,255,0.7); font-size: 14px; margin: 0;">✨ Curious Kelly — Daily learning for lifelong curiosity</p>
        <p style="color: rgba(255,255,255,0.5); font-size: 12px; margin: 15px 0 0 0;">© ${new Date().getFullYear()} Lesson of the Day PBC</p>
      </td>
    </tr>
  </table>
</body>
</html>`;
}

async function main() {
  console.log(`
╔════════════════════════════════════════════════════════════════╗
║  📚 SPECIAL LESSONS GENERATOR                                  ║
║  Creating watch pages and emails for special lessons           ║
╚════════════════════════════════════════════════════════════════╝
`);

  const specialDir = path.join(process.cwd(), 'public', 'lessons', 'special');
  const watchDir = path.join(process.cwd(), 'public', 'watch', 'special');
  const emailDir = path.join(process.cwd(), 'generated-emails', 'special');

  // Ensure directories exist
  if (!fs.existsSync(watchDir)) fs.mkdirSync(watchDir, { recursive: true });
  if (!fs.existsSync(emailDir)) fs.mkdirSync(emailDir, { recursive: true });

  // Get all special lesson files
  const files = fs.readdirSync(specialDir).filter(f => f.endsWith('.json'));
  console.log(`Found ${files.length} special lessons\n`);

  let processed = 0;
  for (const file of files) {
    const lesson = loadSpecialLesson(file);
    if (!lesson) {
      console.log(`⚠️  Could not load ${file}`);
      continue;
    }

    const id = lesson.meta.id.toLowerCase();
    console.log(`━━━ ${lesson.meta.id} ━━━`);
    console.log(`   📚 ${lesson.meta.emoji || '✨'} ${lesson.meta.topic}`);

    // Generate watch page
    const watchPath = path.join(watchDir, `${id}.html`);
    fs.writeFileSync(watchPath, generateWatchPage(lesson, file));
    console.log(`   ✅ Watch page created`);

    // Generate email
    const emailPath = path.join(emailDir, `${id}-email.html`);
    fs.writeFileSync(emailPath, generateEmail(lesson));
    console.log(`   ✅ Email HTML created`);

    processed++;
    console.log('');
  }

  console.log(`════════════════════════════════════════════════════════════════`);
  console.log(`✅ COMPLETE: ${processed} special lessons processed`);
  console.log(`════════════════════════════════════════════════════════════════`);
}

main().catch(console.error);
