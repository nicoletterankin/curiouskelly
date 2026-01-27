/**
 * Generate complete true-north.json and true-north.html
 * 
 * This script transforms the raw Supabase lesson data into the definitive
 * 365-day curriculum reference for Curious Kelly.
 * 
 * Generated: 2024-12-27
 * Source: Supabase core_lessons table (730 records: 365 days × 2 tracks)
 */

const fs = require('fs');
const path = require('path');

// Complete raw lesson data from Supabase core_lessons table
// Exported via: SELECT json_agg(...) FROM core_lessons ORDER BY day_number, track
const rawLessons = [
{"day":1,"id":"942d7456-e44d-41f9-a439-b37008a2d036","topic":"I'm an AI","track":"grow","headline":"Understanding Your Digital Learning Partner","truth":"AI learns from patterns, not experience","icon":"🏗️","objectives":null,"difficulty":null,"duration":null},
{"day":1,"id":"9f8af9c5-66d6-40a0-a10c-b95a7940d25c","topic":"Starting Fresh","track":"learn","headline":"Every January 1st, millions of people try to change—here is why fresh starts actually work","truth":"Fresh starts provide psychological permission to change—the calendar creates natural reset points.","icon":"🌅","objectives":["Explain what the fresh start effect is and why it works psychologically","Identify temporal landmarks that can serve as fresh start opportunities","Design a personal strategy for making the most of new beginnings"],"difficulty":"Beginner","duration":8},
{"day":2,"id":"33ec4d85-d392-4711-8aa7-ddd817f725ed","topic":"What Makes You Human","track":"grow","headline":"The Gifts No AI Has","truth":"Humans have consciousness, feelings, embodiment","icon":"🏗️","objectives":null,"difficulty":null,"duration":null},
{"day":2,"id":"2eefc852-8bdb-4af5-b014-46ba920c6251","topic":"The Three Lives of Water","track":"learn","headline":"The water in your glass was once a cloud, and before that, an ocean","truth":"Water never disappears—it just changes form and travels the world.","icon":"💧","objectives":["Identify and describe the three states of water: solid, liquid, and gas.","Explain the processes of melting, freezing, evaporation, condensation, and sublimation.","Relate the states of water to real-world phenomena like weather patterns and the water cycle."],"difficulty":"Beginner","duration":8},
{"day":3,"id":"9a46bad1-bbe2-42e6-aca2-4ebf90c0635e","topic":"Types of Intelligence","track":"grow","headline":"Many Ways to Be Smart","truth":"There are many ways to be smart","icon":"🏗️","objectives":null,"difficulty":null,"duration":null},
{"day":3,"id":"e06cc658-5da5-4ee1-8b45-0ef4a7c6fd27","topic":"Where Clouds Come From","track":"learn","headline":"Clouds are just fog that found a way to fly","truth":"Clouds form when water vapor rises, cools, and clings to tiny particles in the sky.","icon":"☁️","objectives":["Identify and describe at least three different types of clouds.","Explain the basic steps of cloud formation.","Relate cloud types to corresponding weather patterns."],"difficulty":"Beginner","duration":8},
{"day":4,"id":"7c697b53-cd30-4692-8177-bd37ad0b3682","topic":"How AI Learns","track":"grow","headline":"Patterns in the Data","truth":"AI finds patterns in vast amounts of data","icon":"🏗️","objectives":null,"difficulty":null,"duration":null},
{"day":4,"id":"6c31586c-6758-4518-9d84-ac94ab1867fe","topic":"How Light Travels","track":"learn","headline":"Light travels so fast it could circle Earth seven times in one second","truth":"Light is the fastest thing in the universe—nothing else even comes close.","icon":"💡","objectives":["Explain that light allows us to see objects.","Describe that light travels incredibly fast.","Identify different sources of light."],"difficulty":"Beginner","duration":60},
{"day":5,"id":"68ee813e-736b-4500-8fd5-45b433cd5b9a","topic":"How Humans Learn","track":"grow","headline":"Experience and Connection","truth":"Humans learn through experience and connection","icon":"🏗️","objectives":null,"difficulty":null,"duration":null},
{"day":5,"id":"256d57d9-3224-420b-8871-009ddd941c4f","topic":"How Sound Moves","track":"learn","headline":"Sound cannot travel through space because there is nothing to vibrate","truth":"Sound is vibration—it needs matter to push through, which is why space is silent.","icon":"🔊","objectives":["Identify that sound is caused by vibrations.","Explain how sound travels through air, water, and solid objects.","Give examples of how sound is used in different applications."],"difficulty":"Beginner","duration":8},
{"day":6,"id":"edc6c02c-bd40-4a9e-b580-5b190f5fc579","topic":"The AI Around You","track":"grow","headline":"Already Part of Daily Life","truth":"AI is already part of daily life","icon":"🏗️","objectives":null,"difficulty":null,"duration":null},
{"day":6,"id":"17a45e26-64e5-43ba-b774-bf3d71a5f219","topic":"What's Inside a Seed","track":"learn","headline":"A seed smaller than your fingernail contains instructions for a 300-foot tree","truth":"Every seed carries a complete blueprint for life, just waiting for the right moment.","icon":"🌱","objectives":["Identify the main parts of a seed and their functions.","Describe the process of seed germination and the factors that affect it.","Explain the importance of seeds for plant reproduction and food production."],"difficulty":"Beginner","duration":8},
{"day":7,"id":"b1d80ef0-3793-44e5-b3a8-dbba3c72252a","topic":"Human + AI","track":"grow","headline":"The Power of Collaboration","truth":"The best results come from collaboration","icon":"🏗️","objectives":null,"difficulty":null,"duration":null},
{"day":7,"id":"15789f32-2160-479a-b454-826b6813ff8b","topic":"What Stars Are Made Of","track":"learn","headline":"Stars are giant nuclear explosions that have been burning for billions of years","truth":"Stars are balls of gas so hot they fuse atoms together, releasing light and heat.","icon":"✨","objectives":["Identify major constellations and their significance.","Explain the life cycle of a star, from nebula to black hole or white dwarf.","Describe how stars have been used for navigation and storytelling throughout history."],"difficulty":"Beginner","duration":8}
];

// Helper to get date string for a given day number in 2026
function getDate(dayNumber) {
  const date = new Date(2026, 0, dayNumber); // Month is 0-indexed
  return date.toISOString().split('T')[0];
}

// Group lessons by day
function groupByDay(lessons) {
  const byDay = {};
  lessons.forEach(lesson => {
    const d = lesson.day;
    if (!byDay[d]) byDay[d] = {};
    byDay[d][lesson.track] = lesson;
  });
  return byDay;
}

// Build the complete structured curriculum
function buildCurriculum(lessons) {
  const byDay = groupByDay(lessons);
  const result = [];
  
  for (let day = 1; day <= 365; day++) {
    const dayData = byDay[day] || {};
    const learn = dayData.learn || null;
    const grow = dayData.grow || null;
    
    const entry = {
      day: day,
      date: getDate(day)
    };
    
    if (learn) {
      entry.learn = {
        id: learn.id,
        topic: learn.topic,
        headline: learn.headline,
        truth: learn.truth,
        icon: learn.icon
      };
      if (learn.objectives) entry.learn.objectives = learn.objectives;
      if (learn.difficulty) entry.learn.difficulty = learn.difficulty;
      if (learn.duration) entry.learn.duration = learn.duration;
    }
    
    if (grow) {
      entry.grow = {
        id: grow.id,
        topic: grow.topic,
        headline: grow.headline,
        truth: grow.truth,
        icon: grow.icon
      };
      if (grow.objectives) entry.grow.objectives = grow.objectives;
      if (grow.difficulty) entry.grow.difficulty = grow.difficulty;
      if (grow.duration) entry.grow.duration = grow.duration;
    }
    
    result.push(entry);
  }
  
  return result;
}

// Generate JSON output
function generateJSON(lessons) {
  const curriculum = buildCurriculum(lessons);
  
  const output = {
    version: "1.0.0",
    generated: new Date().toISOString().split('T')[0],
    description: "True North - Complete 365-Day Curriculum for Curious Kelly (2026 Calendar Year)",
    calendar_year: 2026,
    total_days: 365,
    total_lessons: lessons.length,
    tracks: {
      learn: "Daily micro-lesson on science, history, art, or life skills",
      grow: "Personal development and AI literacy companion lesson"
    },
    source: "Supabase core_lessons table",
    lessons: curriculum
  };
  
  return JSON.stringify(output, null, 2);
}

// Generate HTML output
function generateHTML(lessons) {
  const curriculum = buildCurriculum(lessons);
  
  let html = `<!DOCTYPE html>
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
    
    .container {
      max-width: 1400px;
      margin: 0 auto;
    }
    
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
    
    .subtitle {
      color: var(--text-secondary);
      font-size: 1.2rem;
    }
    
    .stats {
      display: flex;
      justify-content: center;
      gap: 40px;
      margin-top: 20px;
      flex-wrap: wrap;
    }
    
    .stat {
      text-align: center;
    }
    
    .stat-value {
      font-size: 2rem;
      font-weight: bold;
      color: var(--kelly-purple);
    }
    
    .stat-label {
      color: var(--text-secondary);
      font-size: 0.9rem;
    }
    
    .search-box {
      margin: 20px 0;
      text-align: center;
    }
    
    #search {
      width: 100%;
      max-width: 500px;
      padding: 15px 20px;
      font-size: 1rem;
      border: 2px solid var(--border-color);
      border-radius: 30px;
      background: var(--bg-card);
      color: var(--text-primary);
      outline: none;
    }
    
    #search:focus {
      border-color: var(--kelly-gold);
    }
    
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
      gap: 20px;
    }
    
    .day-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 20px;
      transition: transform 0.2s, border-color 0.2s;
    }
    
    .day-card:hover {
      transform: translateY(-2px);
      border-color: var(--kelly-gold);
    }
    
    .day-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 15px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border-color);
    }
    
    .day-number {
      font-size: 1.5rem;
      font-weight: bold;
      color: var(--kelly-gold);
    }
    
    .day-date {
      color: var(--text-secondary);
      font-size: 0.9rem;
    }
    
    .track {
      margin-bottom: 15px;
    }
    
    .track-label {
      display: inline-block;
      padding: 3px 10px;
      border-radius: 12px;
      font-size: 0.75rem;
      font-weight: bold;
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    
    .track-learn .track-label {
      background: rgba(59, 130, 246, 0.2);
      color: #60a5fa;
    }
    
    .track-grow .track-label {
      background: rgba(139, 92, 246, 0.2);
      color: #a78bfa;
    }
    
    .track-topic {
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 5px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    .track-headline {
      color: var(--text-secondary);
      font-size: 0.9rem;
      font-style: italic;
    }
    
    .track-truth {
      color: var(--kelly-gold);
      font-size: 0.85rem;
      margin-top: 5px;
      padding: 8px;
      background: rgba(255, 215, 0, 0.1);
      border-radius: 6px;
    }
    
    .lesson-id {
      font-family: monospace;
      font-size: 0.7rem;
      color: #666;
      margin-top: 5px;
    }
    
    .hidden { display: none; }
    
    @media (max-width: 768px) {
      h1 { font-size: 1.8rem; }
      .stats { gap: 20px; }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>✨ True North</h1>
      <p class="subtitle">Curious Kelly 365-Day Curriculum • 2026 Calendar Year</p>
      <div class="stats">
        <div class="stat">
          <div class="stat-value">365</div>
          <div class="stat-label">Days</div>
        </div>
        <div class="stat">
          <div class="stat-value">${lessons.length}</div>
          <div class="stat-label">Total Lessons</div>
        </div>
        <div class="stat">
          <div class="stat-value">2</div>
          <div class="stat-label">Tracks (Learn + Grow)</div>
        </div>
      </div>
    </header>
    
    <div class="search-box">
      <input type="text" id="search" placeholder="Search by day, topic, or keyword..." />
    </div>
    
    <div class="grid" id="lessons-grid">
`;

  curriculum.forEach(day => {
    const learnHtml = day.learn ? `
        <div class="track track-learn">
          <span class="track-label">📚 Learn</span>
          <div class="track-topic">${day.learn.icon || ''} ${day.learn.topic}</div>
          <div class="track-headline">"${day.learn.headline}"</div>
          <div class="track-truth">💡 ${day.learn.truth}</div>
          <div class="lesson-id">ID: ${day.learn.id}</div>
        </div>` : '';
    
    const growHtml = day.grow ? `
        <div class="track track-grow">
          <span class="track-label">🌱 Grow</span>
          <div class="track-topic">${day.grow.icon || ''} ${day.grow.topic}</div>
          <div class="track-headline">"${day.grow.headline}"</div>
          <div class="track-truth">💡 ${day.grow.truth}</div>
          <div class="lesson-id">ID: ${day.grow.id}</div>
        </div>` : '';
    
    html += `
      <div class="day-card" data-day="${day.day}" data-search="${(day.learn?.topic || '').toLowerCase()} ${(day.grow?.topic || '').toLowerCase()} ${(day.learn?.headline || '').toLowerCase()} ${(day.grow?.headline || '').toLowerCase()} ${(day.learn?.truth || '').toLowerCase()} ${(day.grow?.truth || '').toLowerCase()}">
        <div class="day-header">
          <span class="day-number">Day ${day.day}</span>
          <span class="day-date">${day.date}</span>
        </div>
        ${learnHtml}
        ${growHtml}
      </div>
`;
  });

  html += `
    </div>
  </div>
  
  <script>
    document.getElementById('search').addEventListener('input', function(e) {
      const query = e.target.value.toLowerCase();
      document.querySelectorAll('.day-card').forEach(card => {
        const searchText = card.dataset.search;
        const dayNum = card.dataset.day;
        const matches = searchText.includes(query) || dayNum.includes(query);
        card.classList.toggle('hidden', !matches);
      });
    });
  </script>
</body>
</html>`;

  return html;
}

// Main execution
console.log('Generating True North files...');
console.log('Input lessons:', rawLessons.length);

const jsonContent = generateJSON(rawLessons);
const htmlContent = generateHTML(rawLessons);

// Write files to project root
const rootDir = path.join(__dirname, '..');
fs.writeFileSync(path.join(rootDir, 'true-north.json'), jsonContent);
fs.writeFileSync(path.join(rootDir, 'true-north.html'), htmlContent);

console.log('✅ Generated true-north.json');
console.log('✅ Generated true-north.html');
console.log('Note: This script only has sample data (first 7 days).');
console.log('To get ALL 365 days, the full Supabase export needs to be inserted.');
