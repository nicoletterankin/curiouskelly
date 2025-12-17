/**
 * Extract lesson content from day packs for email delivery
 * Generates email-ready HTML from the phase scripts
 * 
 * Usage: node scripts/extract-lesson-for-email.js <dayNumber>
 * Example: node scripts/extract-lesson-for-email.js 17
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Email template using Daily Duo format (Learn + Grow)
function generateEmailHtml(dayNumber, learnData, growData) {
  const dateStr = getDayDate(dayNumber);
  
  const learnSection = learnData ? `
    <tr>
      <td style="padding: 24px; background: linear-gradient(135deg, #fef3c7 0%, #fef9c3 100%); border-radius: 12px;">
        <h2 style="margin: 0 0 16px 0; color: #92400e; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">
          🌟 TODAY'S LEARN LESSON
        </h2>
        <h3 style="margin: 0 0 12px 0; color: #1f2937; font-size: 22px; font-weight: 600;">
          ${learnData.lesson?.topic || 'Daily Lesson'}
        </h3>
        <p style="margin: 0 0 16px 0; color: #374151; font-size: 16px; line-height: 1.6; font-style: italic;">
          ${learnData.lesson?.headline || ''}
        </p>
        ${formatLessonContent(learnData)}
      </td>
    </tr>
  ` : '';

  const growSection = growData ? `
    <tr>
      <td style="padding: 24px; background: linear-gradient(135deg, #ede9fe 0%, #f3e8ff 100%); border-radius: 12px; margin-top: 16px;">
        <h2 style="margin: 0 0 16px 0; color: #5b21b6; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">
          🧠 TODAY'S GROW LESSON
        </h2>
        <h3 style="margin: 0 0 12px 0; color: #1f2937; font-size: 22px; font-weight: 600;">
          ${growData.lesson?.topic || 'Daily Skill'}
        </h3>
        ${formatLessonContent(growData)}
      </td>
    </tr>
  ` : '';

  const wisdomQuote = learnData?.atoms?.find(a => a.phase === 'Wisdom')?.content?.script || 
                       'Stay curious. Keep learning.';

  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Day ${dayNumber}: ${learnData?.lesson?.topic || 'Daily Lesson'} | Curious Kelly</title>
</head>
<body style="margin: 0; padding: 0; background: #f3f4f6; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background: #f3f4f6;">
    <tr>
      <td align="center" style="padding: 24px 16px;">
        <table width="100%" cellpadding="0" cellspacing="0" style="max-width: 600px; background: #ffffff; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
          
          <!-- Header -->
          <tr>
            <td style="padding: 24px; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); text-align: center;">
              <h1 style="margin: 0; color: white; font-size: 28px; font-weight: 400;">
                ✨ Curious Kelly
              </h1>
              <p style="margin: 8px 0 0 0; color: rgba(255,255,255,0.9); font-size: 16px;">
                Day ${dayNumber} — ${dateStr}
              </p>
            </td>
          </tr>

          <!-- Learn Section -->
          ${learnSection}

          <!-- Spacer -->
          <tr><td style="height: 16px;"></td></tr>

          <!-- Grow Section -->
          ${growSection}

          <!-- Wisdom -->
          <tr>
            <td style="padding: 24px; text-align: center; background: #1f2937;">
              <p style="margin: 0 0 8px 0; color: #9ca3af; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">
                ✨ DAILY WISDOM
              </p>
              <p style="margin: 0; color: #f9fafb; font-size: 18px; font-style: italic; line-height: 1.5;">
                "${extractWisdomQuote(wisdomQuote)}"
              </p>
            </td>
          </tr>

          <!-- CTA -->
          <tr>
            <td style="padding: 24px; text-align: center;">
              <a href="https://curiouskelly.com/learn.html?day=${dayNumber}" 
                 style="display: inline-block; padding: 14px 28px; background: #3b82f6; color: white; text-decoration: none; font-size: 16px; font-weight: 600; border-radius: 8px;">
                Experience with Kelly →
              </a>
              <p style="margin: 12px 0 0 0; color: #6b7280; font-size: 14px;">
                Watch today's lesson with voice & video
              </p>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="padding: 16px 24px; background: #f9fafb; text-align: center; border-top: 1px solid #e5e7eb;">
              <p style="margin: 0 0 8px 0; color: #6b7280; font-size: 14px;">
                Stay curious! ✨ — Kelly
              </p>
              <p style="margin: 0; color: #9ca3af; font-size: 12px;">
                <a href="https://curiouskelly.com/settings" style="color: #9ca3af;">Unsubscribe</a> · 
                <a href="https://curiouskelly.com/settings" style="color: #9ca3af;">Preferences</a> · 
                <a href="https://curiouskelly.com" style="color: #9ca3af;">curiouskelly.com</a>
              </p>
              <p style="margin: 8px 0 0 0; color: #d1d5db; font-size: 11px;">
                Lesson of the Day PBC · hello@curiouskelly.com
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>`;
}

function formatLessonContent(data) {
  if (!data?.atoms) return '';
  
  const phases = ['Hook', 'Fact1', 'Fact2', 'Fact3'];
  let html = '';
  
  for (const phaseName of phases) {
    const atom = data.atoms.find(a => a.phase === phaseName);
    if (atom?.content?.script) {
      html += `<p style="margin: 0 0 16px 0; color: #374151; font-size: 16px; line-height: 1.7;">
        ${atom.content.script}
      </p>`;
    }
  }
  
  // Add question if present
  const cliff = data.atoms.find(a => a.phase === 'Cliff');
  if (cliff?.content?.cliffPrompt) {
    html += `<div style="padding: 16px; background: rgba(255,255,255,0.5); border-radius: 8px; border-left: 4px solid #3b82f6;">
      <p style="margin: 0; color: #1f2937; font-size: 16px; font-weight: 500;">
        🤔 ${cliff.content.cliffPrompt}
      </p>
    </div>`;
  }
  
  return html;
}

function extractWisdomQuote(script) {
  // Extract just the key wisdom, removing "So here's today's wisdom:" prefix
  const cleaned = script
    .replace(/^(So here's today's wisdom:|Here's the wisdom:|The wisdom:)\s*/i, '')
    .replace(/^(The takeaway:|Bottom line:)\s*/i, '')
    .trim();
  
  // Take first 2 sentences max
  const sentences = cleaned.match(/[^.!?]+[.!?]+/g) || [cleaned];
  return sentences.slice(0, 2).join(' ').trim();
}

function getDayDate(dayNumber) {
  const months = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December'];
  const monthDays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  
  let remaining = dayNumber;
  let monthIndex = 0;
  
  while (remaining > monthDays[monthIndex]) {
    remaining -= monthDays[monthIndex];
    monthIndex++;
  }
  
  return `${months[monthIndex]} ${remaining}`;
}

// Load day pack data
function loadDayPack(dayNumber) {
  const paddedDay = String(dayNumber).padStart(3, '0');
  const packPath = path.join(__dirname, '..', 'public', 'data', `day-${paddedDay}-complete.js`);
  
  if (!fs.existsSync(packPath)) {
    console.warn(`Day pack not found: ${packPath}`);
    return null;
  }
  
  const content = fs.readFileSync(packPath, 'utf8');
  
  // Extract the JSON object from the JS file
  // Match from the opening { to the closing }; before the LOCAL_PACKS line
  const match = content.match(/window\.CURIOUS_KELLY\.DAY_\d+ = ({[\s\S]+?});\s*(?:\/\/|window|$)/);
  if (!match) {
    console.warn(`Could not parse day pack: ${packPath}`);
    return null;
  }
  
  try {
    return JSON.parse(match[1]);
  } catch (e) {
    console.warn(`JSON parse error for day ${dayNumber}:`, e.message);
    return null;
  }
}

// Load curriculum data for Grow track
async function loadGrowData(dayNumber) {
  const months = ['january', 'february', 'march', 'april', 'may', 'june',
                  'july', 'august', 'september', 'october', 'november', 'december'];
  const monthDays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  
  let remaining = dayNumber;
  let monthIndex = 0;
  
  while (remaining > monthDays[monthIndex]) {
    remaining -= monthDays[monthIndex];
    monthIndex++;
  }
  
  const currPath = path.join(__dirname, '..', 'public', 'data', 'curriculum', 
                             'year2-ai-fluency', `${months[monthIndex]}_curriculum.json`);
  
  if (!fs.existsSync(currPath)) return null;
  
  try {
    const data = JSON.parse(fs.readFileSync(currPath, 'utf8'));
    const day = data.days?.find(d => d.day === dayNumber);
    if (!day) return null;
    
    return {
      lesson: {
        topic: day.title,
        headline: day.learning_objective
      },
      atoms: [] // Grow track doesn't have full atoms yet
    };
  } catch (e) {
    return null;
  }
}

// Main
async function main() {
  const dayNumber = parseInt(process.argv[2] || '17');
  
  console.log(`Extracting email content for Day ${dayNumber}...`);
  
  const learnData = loadDayPack(dayNumber);
  const growData = await loadGrowData(dayNumber);
  
  if (!learnData && !growData) {
    console.error(`No data found for Day ${dayNumber}`);
    process.exit(1);
  }
  
  const html = generateEmailHtml(dayNumber, learnData, growData);
  
  // Write to output
  const outputDir = path.join(__dirname, '..', 'generated-emails');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  const outputPath = path.join(outputDir, `day-${String(dayNumber).padStart(3, '0')}-email.html`);
  fs.writeFileSync(outputPath, html);
  
  console.log(`Email HTML written to: ${outputPath}`);
  console.log(`Learn topic: ${learnData?.lesson?.topic || 'N/A'}`);
  console.log(`Grow topic: ${growData?.lesson?.topic || 'N/A'}`);
}

main().catch(console.error);
