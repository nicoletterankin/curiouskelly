#!/usr/bin/env npx tsx
/**
 * YEAR 2 (GROW TRACK) BATCH GENERATOR
 * 
 * Generates lessons, watch pages, and emails for Year 2 AI Fluency curriculum
 * 
 * Usage:
 *   npx tsx scripts/generate-year2-batch.ts --range=1-10
 *   npx tsx scripts/generate-year2-batch.ts --range=1-50
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';

interface CurriculumDay {
  day: number;
  date: string;
  title: string;
  learning_objective: string;
  icon?: string;
}

interface CurriculumMonth {
  year: number;
  program: string;
  month: string;
  theme: string;
  themeDescription: string;
  days: CurriculumDay[];
}

function getMonthForDay(day: number): string {
  const monthDays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  const months = ['january', 'february', 'march', 'april', 'may', 'june',
                  'july', 'august', 'september', 'october', 'november', 'december'];
  
  let remaining = day;
  let monthIndex = 0;
  
  while (remaining > monthDays[monthIndex]) {
    remaining -= monthDays[monthIndex];
    monthIndex++;
  }
  
  return months[monthIndex];
}

function getDateForDay(day: number): { month: string; dayOfMonth: number; monthNum: number } {
  const monthDays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  const months = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December'];
  
  let remaining = day;
  let monthIndex = 0;
  
  while (remaining > monthDays[monthIndex]) {
    remaining -= monthDays[monthIndex];
    monthIndex++;
  }
  
  return { month: months[monthIndex], dayOfMonth: remaining, monthNum: monthIndex + 1 };
}

function loadCurriculum(month: string): CurriculumMonth | null {
  const curriculumPath = path.join(
    process.cwd(), 
    'public', 
    'data', 
    'curriculum', 
    'year2-ai-fluency', 
    `${month}_curriculum.json`
  );
  
  if (!fs.existsSync(curriculumPath)) {
    console.log(`⚠️  Curriculum not found: ${curriculumPath}`);
    return null;
  }
  
  return JSON.parse(fs.readFileSync(curriculumPath, 'utf-8'));
}

function getCurriculumDay(day: number): CurriculumDay | null {
  const month = getMonthForDay(day);
  const curriculum = loadCurriculum(month);
  
  if (!curriculum) return null;
  
  return curriculum.days.find(d => d.day === day) || null;
}

function generateLesson(currDay: CurriculumDay): object {
  const { day, title, learning_objective, icon } = currDay;
  const date = getDateForDay(day);
  
  // Generate content based on the title and learning objective
  const emoji = icon || '🤖';
  
  return {
    meta: {
      year: 2,
      track: "grow",
      day: day,
      date: `2026-${String(date.monthNum).padStart(2, '0')}-${String(date.dayOfMonth).padStart(2, '0')}`,
      topic: title,
      emoji: emoji,
      category: "AI Fluency",
      version: "v4.0-launch-locked",
      target_audience: "adult",
      voice_id: "wAdymQH5YucAkXwmrdL0"
    },
    headline: learning_objective,
    universal_truth: `Understanding ${title.toLowerCase()} empowers you to learn and grow more effectively`,
    fun_facts: [
      `AI fluency is becoming as essential as digital literacy was a generation ago`,
      `Meta-learning—learning how to learn—multiplies all other learning`,
      `The most effective learners understand both their tools and themselves`
    ],
    discussion_questions: [
      `How does today's topic change how you think about learning?`,
      `What would mastering this skill enable you to do?`,
      `How can you apply this insight immediately?`
    ],
    phases: {
      hook: {
        script: `Welcome to Day ${day} of your AI fluency journey. Today we explore ${title}. ${learning_objective.split('.')[0]}.`,
        duration: 12
      },
      cliff: {
        script: `Here's the key question: Why does understanding ${title.toLowerCase()} matter for your growth as a learner?`,
        prompt: `Why does this matter?`,
        options: [
          {
            text: `${learning_objective}`,
            letter: "A",
            quality: "best",
            response: `Exactly. This understanding transforms how you approach learning and using AI as a tool.`
          },
          {
            text: `It's just interesting information`,
            letter: "B",
            quality: "good",
            response: `It's more than interesting—it's actionable. Let me show you why...`
          }
        ],
        duration: 14
      },
      fact1: {
        title: "The Core Insight",
        script: `${learning_objective} This isn't abstract—it directly affects how effectively you can learn and grow.`,
        duration: 14
      },
      fact2: {
        title: "Practical Application",
        script: `Here's how to apply this: When you understand ${title.toLowerCase()}, you make better decisions about when and how to use AI in your learning journey.`,
        duration: 14
      },
      fact3: {
        title: "Building the Skill",
        script: `This knowledge compounds. Each day's insight connects to the others, building a complete picture of AI fluency and meta-learning.`,
        duration: 12
      },
      wisdom: {
        script: `Here's today's wisdom: ${learning_objective.split('.')[0]}. Carry this forward as you continue growing.`,
        duration: 10
      },
      outro: {
        script: `That's Day ${day} of the GROW track. Tomorrow brings another step in your AI fluency journey. Until then, practice what you've learned today.`,
        duration: 10
      }
    },
    phaseOrder: ["hook", "cliff", "fact1", "fact2", "fact3", "wisdom", "outro"],
    totalDuration: 86,
    learnTrack: {
      title: "Cross-Reference Activity",
      emoji: "🔗",
      learning_objective: "Connect today's AI learning to broader knowledge",
      activity: `Think about how today's topic—${title}—relates to something you learned in the LEARN track. How do these insights complement each other?`
    }
  };
}

function generateWatchPage(lesson: any): string {
  const { day, topic, emoji } = lesson.meta;
  const date = getDateForDay(day);
  const headline = lesson.headline || '';
  
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GROW Day ${day}: ${topic} — Curious Kelly</title>
  <meta name="description" content="${headline}">
  
  <meta property="og:title" content="GROW Day ${day}: ${topic}">
  <meta property="og:description" content="${headline}">
  <meta property="og:type" content="video.other">
  <meta property="og:video" content="https://videos.curiouskelly.com/year2/day-${String(day).padStart(3, '0')}.mp4">
  <meta property="og:image" content="https://videos.curiouskelly.com/year2/day-${String(day).padStart(3, '0')}-thumb.jpg">
  
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: linear-gradient(135deg, #0f3460 0%, #16213e 50%, #1a1a2e 100%);
      min-height: 100vh;
      color: white;
    }
    .container { max-width: 900px; margin: 0 auto; padding: 40px 20px; }
    header { text-align: center; margin-bottom: 30px; }
    .track-badge {
      display: inline-block;
      background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
      color: white;
      padding: 6px 16px;
      border-radius: 20px;
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 15px;
    }
    .day-info { color: rgba(255,255,255,0.7); font-size: 0.95rem; margin-bottom: 10px; }
    h1 { font-size: 2rem; margin-bottom: 10px; }
    .emoji { font-size: 2.5rem; margin-bottom: 10px; display: block; }
    .headline { font-size: 1.1rem; opacity: 0.9; max-width: 600px; margin: 0 auto; line-height: 1.5; }
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
    video { width: 100%; display: block; }
    .content-section {
      background: rgba(255,255,255,0.05);
      border-radius: 16px;
      padding: 30px;
      margin: 30px auto;
      max-width: 800px;
    }
    .content-section h2 { font-size: 1.3rem; margin-bottom: 20px; color: #00cec9; }
    .fun-facts { list-style: none; }
    .fun-facts li {
      padding: 12px 0;
      border-bottom: 1px solid rgba(255,255,255,0.1);
      display: flex;
      gap: 12px;
    }
    .fun-facts li:last-child { border-bottom: none; }
    .fun-facts li::before { content: "🤖"; }
    .learn-track {
      background: linear-gradient(135deg, rgba(0,184,148,0.2) 0%, rgba(0,206,201,0.1) 100%);
      border: 1px solid rgba(0,184,148,0.3);
    }
    .learn-track h2 { color: #00b894; }
    .activity-text { line-height: 1.7; font-size: 1.05rem; }
    .nav-section { display: flex; justify-content: space-between; margin-top: 40px; gap: 20px; }
    .nav-button {
      flex: 1;
      display: block;
      background: rgba(255,255,255,0.1);
      color: white;
      padding: 16px 20px;
      border-radius: 12px;
      text-decoration: none;
      text-align: center;
      transition: background 0.2s;
    }
    .nav-button:hover { background: rgba(255,255,255,0.2); }
    footer {
      text-align: center;
      margin-top: 60px;
      padding: 20px;
      opacity: 0.7;
      font-size: 0.9rem;
    }
    footer a { color: #00cec9; text-decoration: none; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <span class="track-badge">GROW Track — Year 2</span>
      <p class="day-info">${date.month} ${date.dayOfMonth} — Day ${day} of 365</p>
      <span class="emoji">${emoji || '🤖'}</span>
      <h1>${topic}</h1>
      <p class="headline">${headline}</p>
    </header>
    
    <div class="video-container">
      <video controls poster="https://videos.curiouskelly.com/year2/day-${String(day).padStart(3, '0')}-thumb.jpg">
        <source src="https://videos.curiouskelly.com/year2/day-${String(day).padStart(3, '0')}.mp4" type="video/mp4">
        Your browser does not support video playback.
      </video>
    </div>
    
    <section class="content-section">
      <h2>Key Insights</h2>
      <ul class="fun-facts">
        ${lesson.fun_facts.map((fact: string) => `<li>${fact}</li>`).join('\n        ')}
      </ul>
    </section>
    
    ${lesson.learnTrack ? `
    <section class="content-section learn-track">
      <h2>🔗 ${lesson.learnTrack.title}</h2>
      <p class="activity-text">${lesson.learnTrack.activity}</p>
    </section>
    ` : ''}
    
    <div class="nav-section">
      <a href="/watch/year2/day-${String(day - 1).padStart(3, '0')}.html" class="nav-button">← Previous Day</a>
      <a href="/calendar" class="nav-button">Calendar</a>
      <a href="/watch/year2/day-${String(day + 1).padStart(3, '0')}.html" class="nav-button">Next Day →</a>
    </div>
    
    <footer>
      <p>🤖 <a href="/">Curious Kelly</a> — GROW Track: AI Fluency & Meta-Learning</p>
      <p style="margin-top: 10px;">© ${new Date().getFullYear()} Lesson of the Day PBC</p>
    </footer>
  </div>
</body>
</html>`;
}

function generateEmail(lesson: any): string {
  const { day, topic, emoji } = lesson.meta;
  const date = getDateForDay(day);
  const headline = lesson.headline || '';
  
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GROW Day ${day}: ${topic}</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5;">
  <table width="100%" cellpadding="0" cellspacing="0" style="max-width: 600px; margin: 0 auto; background: white;">
    <tr>
      <td style="padding: 40px 30px; text-align: center; background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);">
        <p style="font-size: 12px; color: rgba(255,255,255,0.7); margin: 0 0 5px 0; text-transform: uppercase; letter-spacing: 1px;">GROW Track — Year 2</p>
        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 15px 0;">${date.month} ${date.dayOfMonth} — Day ${day}</p>
        <p style="font-size: 48px; margin: 0;">${emoji || '🤖'}</p>
        <h1 style="color: white; font-size: 26px; margin: 15px 0 10px 0;">${topic}</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 15px; margin: 0; line-height: 1.5;">${headline}</p>
      </td>
    </tr>
    <tr>
      <td style="padding: 30px; text-align: center;">
        <a href="https://curiouskelly.com/watch/year2/day-${String(day).padStart(3, '0')}.html" style="display: inline-block;">
          <img src="https://videos.curiouskelly.com/year2/day-${String(day).padStart(3, '0')}-thumb.jpg" alt="Watch GROW Day ${day}" style="width: 100%; max-width: 500px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.15);">
        </a>
        <p style="margin: 20px 0 0 0;">
          <a href="https://curiouskelly.com/watch/year2/day-${String(day).padStart(3, '0')}.html" style="display: inline-block; background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); color: white; padding: 14px 35px; border-radius: 25px; text-decoration: none; font-weight: 600;">Watch Now</a>
        </p>
      </td>
    </tr>
    <tr>
      <td style="padding: 0 30px 30px 30px;">
        <div style="background: #f8f9fa; border-radius: 12px; padding: 25px;">
          <h2 style="margin: 0 0 15px 0; font-size: 18px; color: #333;">🤖 Key Insights</h2>
          ${lesson.fun_facts.map((fact: string) => `<p style="margin: 10px 0; color: #555; line-height: 1.5;">• ${fact}</p>`).join('')}
        </div>
      </td>
    </tr>
    ${lesson.learnTrack ? `
    <tr>
      <td style="padding: 0 30px 30px 30px;">
        <div style="background: linear-gradient(135deg, rgba(0,184,148,0.1) 0%, rgba(0,206,201,0.05) 100%); border: 1px solid rgba(0,184,148,0.2); border-radius: 12px; padding: 25px;">
          <h2 style="margin: 0 0 15px 0; font-size: 18px; color: #00b894;">🔗 ${lesson.learnTrack.title}</h2>
          <p style="margin: 0; color: #555; line-height: 1.6;">${lesson.learnTrack.activity}</p>
        </div>
      </td>
    </tr>
    ` : ''}
    <tr>
      <td style="padding: 30px; background: #0f3460; text-align: center;">
        <p style="color: rgba(255,255,255,0.7); font-size: 14px; margin: 0;">🤖 Curious Kelly — GROW Track: AI Fluency & Meta-Learning</p>
        <p style="color: rgba(255,255,255,0.5); font-size: 12px; margin: 15px 0 0 0;">© ${new Date().getFullYear()} Lesson of the Day PBC</p>
      </td>
    </tr>
  </table>
</body>
</html>`;
}

async function main() {
  const args = process.argv.slice(2);
  let startDay = 1;
  let endDay = 10;
  
  // Parse --range argument
  const rangeArg = args.find(a => a.startsWith('--range='));
  if (rangeArg) {
    const [start, end] = rangeArg.replace('--range=', '').split('-').map(Number);
    startDay = start;
    endDay = end;
  }
  
  console.log(`
╔════════════════════════════════════════════════════════════════╗
║  🤖 YEAR 2 (GROW TRACK) BATCH GENERATOR                       ║
║  Creating lessons, watch pages, and emails                     ║
╚════════════════════════════════════════════════════════════════╝

🗓️  Processing days: ${startDay} to ${endDay}
`);

  const lessonsDir = path.join(process.cwd(), 'public', 'lessons', 'year2');
  const watchDir = path.join(process.cwd(), 'public', 'watch', 'year2');
  const emailDir = path.join(process.cwd(), 'generated-emails', 'year2');

  // Ensure directories exist
  if (!fs.existsSync(lessonsDir)) fs.mkdirSync(lessonsDir, { recursive: true });
  if (!fs.existsSync(watchDir)) fs.mkdirSync(watchDir, { recursive: true });
  if (!fs.existsSync(emailDir)) fs.mkdirSync(emailDir, { recursive: true });

  let processed = 0;
  let skipped = 0;

  for (let day = startDay; day <= endDay; day++) {
    const currDay = getCurriculumDay(day);
    
    if (!currDay) {
      console.log(`⚠️  Day ${day}: No curriculum data found`);
      skipped++;
      continue;
    }

    console.log(`━━━ Day ${day} ━━━`);
    console.log(`   📚 ${currDay.title}`);

    // Generate lesson JSON
    const lesson = generateLesson(currDay);
    const lessonPath = path.join(lessonsDir, `day-${String(day).padStart(3, '0')}.json`);
    fs.writeFileSync(lessonPath, JSON.stringify(lesson, null, 2));
    console.log(`   ✅ Lesson JSON created`);

    // Generate watch page
    const watchPath = path.join(watchDir, `day-${String(day).padStart(3, '0')}.html`);
    fs.writeFileSync(watchPath, generateWatchPage(lesson));
    console.log(`   ✅ Watch page created`);

    // Generate email
    const emailPath = path.join(emailDir, `day-${String(day).padStart(3, '0')}-email.html`);
    fs.writeFileSync(emailPath, generateEmail(lesson));
    console.log(`   ✅ Email HTML created`);

    processed++;
    console.log('');
  }

  console.log(`════════════════════════════════════════════════════════════════`);
  console.log(`✅ BATCH COMPLETE: ${processed} days processed, ${skipped} skipped`);
  console.log(`════════════════════════════════════════════════════════════════`);
}

main().catch(console.error);
