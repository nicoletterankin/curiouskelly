/**
 * Fetch 365-day curriculum from Supabase and generate true-north files
 * 
 * This script connects to Supabase using the project credentials
 * and generates both true-north.json and true-north.html
 * 
 * Usage: node scripts/fetch-and-generate-true-north.js
 */

import fs from 'fs';
import { createRequire } from 'module';
import { config } from 'dotenv';
config();

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.PUBLIC_SUPABASE_ANON_KEY || process.env.SUPABASE_ANON_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.log('Supabase credentials not found in environment.');
  console.log('Using embedded lesson data from previous export...');
  
  // Fall back to reading from existing file
  const existingData = JSON.parse(fs.readFileSync(new URL('./supabase-365.json', import.meta.url)));
  generateFiles(existingData);
  process.exit(0);
}

const url = new URL(SUPABASE_URL);
const hostname = url.hostname;

// Build the query for all lessons joined by day
const query = `
  select=day_number,
    learn:core_lessons!track=eq.learn(id,topic,marketing_headline,universal_truth,icon_emoji),
    grow:core_lessons!track=eq.grow(id,topic,marketing_headline,universal_truth,icon_emoji)
  &order=day_number
`;

// Simple REST query to get pivoted lessons
async function fetchLessons() {
  // Using the REST API to get lessons
  const learnUrl = `${SUPABASE_URL}/rest/v1/core_lessons?select=day_number,id,topic,marketing_headline,universal_truth,icon_emoji&track=eq.learn&order=day_number`;
  const growUrl = `${SUPABASE_URL}/rest/v1/core_lessons?select=day_number,id,topic,marketing_headline,universal_truth,icon_emoji&track=eq.grow&order=day_number`;
  
  console.log('Fetching learn track...');
  const learnData = await fetch(learnUrl, {
    headers: { 'apikey': SUPABASE_KEY, 'Authorization': `Bearer ${SUPABASE_KEY}` }
  }).then(r => r.json());
  
  console.log('Fetching grow track...');
  const growData = await fetch(growUrl, {
    headers: { 'apikey': SUPABASE_KEY, 'Authorization': `Bearer ${SUPABASE_KEY}` }
  }).then(r => r.json());
  
  console.log(`Got ${learnData.length} learn lessons, ${growData.length} grow lessons`);
  
  // Combine by day
  const lessons = [];
  const learnByDay = Object.fromEntries(learnData.map(l => [l.day_number, l]));
  const growByDay = Object.fromEntries(growData.map(g => [g.day_number, g]));
  
  for (let day = 1; day <= 365; day++) {
    const learn = learnByDay[day] || {};
    const grow = growByDay[day] || {};
    
    const date = new Date(2026, 0, day);
    const dateStr = date.toISOString().split('T')[0];
    
    lessons.push({
      day: day,
      date_2026: dateStr,
      learn_id: learn.id || null,
      learn_topic: learn.topic || null,
      learn_headline: learn.marketing_headline || null,
      learn_truth: learn.universal_truth || null,
      learn_icon: learn.icon_emoji || null,
      grow_id: grow.id || null,
      grow_topic: grow.topic || null,
      grow_headline: grow.marketing_headline || null,
      grow_truth: grow.universal_truth || null,
      grow_icon: grow.icon_emoji || null
    });
  }
  
  return lessons;
}

function generateFiles(lessons) {
  console.log(`Generating files with ${lessons.length} days...`);
  
  // Create true-north.json
  const trueNorth = {
    version: "1.0.0",
    generated: new Date().toISOString().split('T')[0],
    description: "True North - Complete 365-Day Curriculum for Curious Kelly (2026 Calendar Year)",
    calendar_year: 2026,
    total_days: lessons.length,
    total_lessons: lessons.length * 2,
    tracks: {
      learn: "Daily micro-lesson on science, history, art, or life skills",
      grow: "Personal development and AI literacy companion lesson"
    },
    source: "Supabase core_lessons table",
    lessons: lessons
  };
  
  fs.writeFileSync('true-north.json', JSON.stringify(trueNorth, null, 2));
  console.log('✅ Created true-north.json');
  
  // Create true-north.html
  const html = generateHTML(lessons);
  fs.writeFileSync('true-north.html', html);
  console.log('✅ Created true-north.html');
  
  // Create true-north.csv
  const csv = generateCSV(lessons);
  fs.writeFileSync('true-north.csv', csv);
  console.log('✅ Created true-north.csv');
  
  console.log('\n📦 All files ready to share!');
}

function generateHTML(lessons) {
  const rows = lessons.map(l => `
    <tr>
      <td class="day">${l.day}</td>
      <td class="date">${l.date_2026}</td>
      <td class="icon">${l.learn_icon || ''}</td>
      <td class="topic">${l.learn_topic || ''}</td>
      <td class="headline">${l.learn_headline || ''}</td>
      <td class="icon">${l.grow_icon || ''}</td>
      <td class="topic">${l.grow_topic || ''}</td>
    </tr>`).join('\n');
  
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>True North - Curious Kelly 365-Day Curriculum (2026)</title>
  <style>
    :root {
      --kelly-gold: #FFD700;
      --kelly-purple: #6B46C1;
      --bg-dark: #0f0f0f;
      --bg-card: #1a1a1a;
      --text-primary: #ffffff;
      --text-secondary: #a0a0a0;
      --border-color: #333;
    }
    
    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body {
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: var(--bg-dark);
      color: var(--text-primary);
      line-height: 1.6;
      padding: 20px;
    }
    
    .container { max-width: 1600px; margin: 0 auto; }
    
    header {
      text-align: center;
      padding: 40px 20px;
      border-bottom: 2px solid var(--kelly-gold);
      margin-bottom: 40px;
    }
    
    h1 {
      font-size: 2.5rem;
      color: var(--kelly-gold);
      margin-bottom: 10px;
    }
    
    .subtitle { color: var(--text-secondary); font-size: 1.2rem; }
    
    .stats {
      display: flex;
      justify-content: center;
      gap: 40px;
      margin-top: 20px;
    }
    
    .stat { text-align: center; }
    .stat-value { font-size: 2rem; color: var(--kelly-gold); font-weight: bold; }
    .stat-label { color: var(--text-secondary); font-size: 0.9rem; }
    
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--bg-card);
      border-radius: 8px;
      overflow: hidden;
    }
    
    th {
      background: var(--kelly-purple);
      color: white;
      padding: 15px 10px;
      text-align: left;
      font-weight: 600;
      position: sticky;
      top: 0;
    }
    
    td { padding: 12px 10px; border-bottom: 1px solid var(--border-color); }
    
    tr:hover { background: rgba(255, 215, 0, 0.1); }
    
    .day { font-weight: bold; color: var(--kelly-gold); width: 50px; text-align: center; }
    .date { color: var(--text-secondary); width: 100px; }
    .icon { font-size: 1.5rem; width: 40px; text-align: center; }
    .topic { font-weight: 500; }
    .headline { color: var(--text-secondary); font-size: 0.9rem; }
    
    .search {
      margin-bottom: 20px;
      display: flex;
      gap: 10px;
    }
    
    input[type="text"] {
      flex: 1;
      padding: 12px 20px;
      border: 2px solid var(--border-color);
      border-radius: 8px;
      background: var(--bg-card);
      color: var(--text-primary);
      font-size: 1rem;
    }
    
    input[type="text"]:focus {
      outline: none;
      border-color: var(--kelly-gold);
    }
    
    .footer {
      text-align: center;
      padding: 40px 20px;
      color: var(--text-secondary);
      margin-top: 40px;
    }
    
    @media (max-width: 1200px) {
      .headline { display: none; }
    }
    
    @media (max-width: 768px) {
      .stats { flex-direction: column; gap: 15px; }
      th, td { padding: 8px 5px; font-size: 0.85rem; }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>✨ True North</h1>
      <p class="subtitle">Complete 365-Day Curriculum for Curious Kelly (2026)</p>
      <div class="stats">
        <div class="stat">
          <div class="stat-value">365</div>
          <div class="stat-label">Days</div>
        </div>
        <div class="stat">
          <div class="stat-value">730</div>
          <div class="stat-label">Lessons</div>
        </div>
        <div class="stat">
          <div class="stat-value">2</div>
          <div class="stat-label">Tracks</div>
        </div>
      </div>
    </header>
    
    <div class="search">
      <input type="text" id="search" placeholder="Search lessons by topic, headline, or day..." />
    </div>
    
    <table id="lessons">
      <thead>
        <tr>
          <th>Day</th>
          <th>Date</th>
          <th colspan="3">Learn Track</th>
          <th colspan="2">Grow Track</th>
        </tr>
      </thead>
      <tbody>
        ${rows}
      </tbody>
    </table>
    
    <footer class="footer">
      <p>Generated ${new Date().toISOString().split('T')[0]} | Curious Kelly by Lesson of the Day PBC</p>
      <p>Data source: Supabase core_lessons table</p>
    </footer>
  </div>
  
  <script>
    document.getElementById('search').addEventListener('input', function(e) {
      const query = e.target.value.toLowerCase();
      const rows = document.querySelectorAll('#lessons tbody tr');
      rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(query) ? '' : 'none';
      });
    });
  </script>
</body>
</html>`;
}

function generateCSV(lessons) {
  const header = 'day,date_2026,learn_id,learn_topic,learn_headline,learn_truth,learn_icon,grow_id,grow_topic,grow_headline,grow_truth,grow_icon';
  const rows = lessons.map(l => {
    const escapeCsv = (s) => s ? '"' + String(s).replace(/"/g, '""') + '"' : '';
    return [
      l.day,
      l.date_2026,
      l.learn_id || '',
      escapeCsv(l.learn_topic),
      escapeCsv(l.learn_headline),
      escapeCsv(l.learn_truth),
      l.learn_icon || '',
      l.grow_id || '',
      escapeCsv(l.grow_topic),
      escapeCsv(l.grow_headline),
      escapeCsv(l.grow_truth),
      l.grow_icon || ''
    ].join(',');
  });
  return header + '\n' + rows.join('\n');
}

// Main execution
if (SUPABASE_URL && SUPABASE_KEY) {
  fetchLessons().then(generateFiles).catch(console.error);
} else {
  console.log('Using local data...');
}
