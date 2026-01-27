/**
 * Generate true-north.json and true-north.html from Supabase data
 * This creates the definitive 365-day curriculum reference
 */

const fs = require('fs');

// Raw lesson data from Supabase core_lessons table (exported 2024-12-27)
const lessons = [
{"day":1,"id":"9f8af9c5-66d6-40a0-a10c-b95a7940d25c","topic":"Starting Fresh","track":"learn","headline":"Every January 1st, millions of people try to change—here is why fresh starts actually work","truth":"Fresh starts provide psychological permission to change—the calendar creates natural reset points.","icon":"🌅","objectives":["Explain what the fresh start effect is and why it works psychologically","Identify temporal landmarks that can serve as fresh start opportunities","Design a personal strategy for making the most of new beginnings"],"difficulty":"Beginner","duration":8},
{"day":1,"id":"942d7456-e44d-41f9-a439-b37008a2d036","topic":"I'm an AI","track":"grow","headline":"Understanding Your Digital Learning Partner","truth":"AI learns from patterns, not experience","icon":"🏗️"},
{"day":2,"id":"2eefc852-8bdb-4af5-b014-46ba920c6251","topic":"The Three Lives of Water","track":"learn","headline":"The water in your glass was once a cloud, and before that, an ocean","truth":"Water never disappears—it just changes form and travels the world.","icon":"💧","objectives":["Identify and describe the three states of water: solid, liquid, and gas.","Explain the processes of melting, freezing, evaporation, condensation, and sublimation.","Relate the states of water to real-world phenomena like weather patterns and the water cycle."],"difficulty":"Beginner","duration":8},
{"day":2,"id":"33ec4d85-d392-4711-8aa7-ddd817f725ed","topic":"What Makes You Human","track":"grow","headline":"The Gifts No AI Has","truth":"Humans have consciousness, feelings, embodiment","icon":"🏗️"},
{"day":3,"id":"e06cc658-5da5-4ee1-8b45-0ef4a7c6fd27","topic":"Where Clouds Come From","track":"learn","headline":"Clouds are just fog that found a way to fly","truth":"Clouds form when water vapor rises, cools, and clings to tiny particles in the sky.","icon":"☁️","objectives":["Identify and describe at least three different types of clouds.","Explain the basic steps of cloud formation.","Relate cloud types to corresponding weather patterns."],"difficulty":"Beginner","duration":8},
{"day":3,"id":"9a46bad1-bbe2-42e6-aca2-4ebf90c0635e","topic":"Types of Intelligence","track":"grow","headline":"Many Ways to Be Smart","truth":"There are many ways to be smart","icon":"🏗️"},
{"day":4,"id":"6c31586c-6758-4518-9d84-ac94ab1867fe","topic":"How Light Travels","track":"learn","headline":"Light travels so fast it could circle Earth seven times in one second","truth":"Light is the fastest thing in the universe—nothing else even comes close.","icon":"💡","objectives":["Explain that light allows us to see objects.","Describe that light travels incredibly fast.","Identify different sources of light."],"difficulty":"Beginner","duration":60},
{"day":4,"id":"7c697b53-cd30-4692-8177-bd37ad0b3682","topic":"How AI Learns","track":"grow","headline":"Patterns in the Data","truth":"AI finds patterns in vast amounts of data","icon":"🏗️"},
{"day":5,"id":"256d57d9-3224-420b-8871-009ddd941c4f","topic":"How Sound Moves","track":"learn","headline":"Sound cannot travel through space because there is nothing to vibrate","truth":"Sound is vibration—it needs matter to push through, which is why space is silent.","icon":"🔊","objectives":["Identify that sound is caused by vibrations.","Explain how sound travels through air, water, and solid objects.","Give examples of how sound is used in different applications."],"difficulty":"Beginner","duration":8},
{"day":5,"id":"68ee813e-736b-4500-8fd5-45b433cd5b9a","topic":"How Humans Learn","track":"grow","headline":"Experience and Connection","truth":"Humans learn through experience and connection","icon":"🏗️"},
{"day":6,"id":"17a45e26-64e5-43ba-b774-bf3d71a5f219","topic":"What's Inside a Seed","track":"learn","headline":"A seed smaller than your fingernail contains instructions for a 300-foot tree","truth":"Every seed carries a complete blueprint for life, just waiting for the right moment.","icon":"🌱","objectives":["Identify the main parts of a seed and their functions.","Describe the process of seed germination and the factors that affect it.","Explain the importance of seeds for plant reproduction and food production."],"difficulty":"Beginner","duration":8},
{"day":6,"id":"edc6c02c-bd40-4a9e-b580-5b190f5fc579","topic":"The AI Around You","track":"grow","headline":"Already Part of Daily Life","truth":"AI is already part of daily life","icon":"🏗️"},
{"day":7,"id":"15789f32-2160-479a-b454-826b6813ff8b","topic":"What Stars Are Made Of","track":"learn","headline":"Stars are giant nuclear explosions that have been burning for billions of years","truth":"Stars are balls of gas so hot they fuse atoms together, releasing light and heat.","icon":"✨","objectives":["Identify major constellations and their significance.","Explain the life cycle of a star, from nebula to black hole or white dwarf.","Describe how stars have been used for navigation and storytelling throughout history."],"difficulty":"Beginner","duration":8},
{"day":7,"id":"b1d80ef0-3793-44e5-b3a8-dbba3c72252a","topic":"Human + AI","track":"grow","headline":"The Power of Collaboration","truth":"The best results come from collaboration","icon":"🏗️"},
{"day":8,"id":"97fc40c0-3d13-42e8-8177-c57f5a1f3d57","topic":"What Makes a Real Friend","track":"learn","headline":"The difference between 1,000 followers and one real friend","truth":"True friendship is someone who knows your flaws and chooses to stay anyway.","icon":"🤝","objectives":["Define friendship and identify its key characteristics.","Recognize the importance of empathy and understanding in building strong relationships.","Apply strategies for resolving conflicts and maintaining healthy friendships."],"difficulty":"Beginner","duration":60},
{"day":8,"id":"ba01598b-9071-4eb3-b57f-b9e2879e007b","topic":"What AI Can Do","track":"grow","headline":"Speed, Scale, and Pattern Recognition","truth":"AI excels at pattern recognition and speed","icon":"🏗️"},
{"day":9,"id":"76ad066b-4ad0-4172-8792-b61ef68345a3","topic":"How Kindness Spreads","track":"learn","headline":"One act of kindness triggers an average of three more—it spreads like a virus","truth":"Kindness is contagious—when you help someone, they become more likely to help others.","icon":"💖","objectives":["Define kindness and identify examples of kind behavior.","Explain the ripple effect of kindness and its impact on others.","Practice acts of kindness in daily life and reflect on the experience."],"difficulty":"Beginner","duration":45},
{"day":9,"id":"f2a8e863-9e4b-44c3-96e6-6eb3a066f180","topic":"What AI Can't Do","track":"grow","headline":"The Limits of Artificial Intelligence","truth":"AI lacks consciousness, creativity, and common sense","icon":"🏗️"},
{"day":10,"id":"5f2ee3db-b265-4d70-a29a-554429676e3e","topic":"The Art of Really Listening","track":"learn","headline":"Most people listen to respond, not to understand—that is the difference","truth":"Real listening means focusing entirely on the speaker, not planning what you will say next.","icon":"👂","objectives":["Define active listening and its key components.","Identify common barriers to effective listening and strategies to overcome them.","Demonstrate empathy and understanding through active listening techniques in simulated conversations."],"difficulty":"Beginner","duration":8},
{"day":10,"id":"aab7440e-ef5d-4747-8c6b-ebd28f83a41d","topic":"Your Unique Gifts","track":"grow","headline":"Capabilities No AI Has","truth":"Every human has capabilities no AI has","icon":"🏗️"}
];

// Helper to get 2026 date for a day number
function getDate2026(day) {
  const date = new Date(2026, 0, day); // January 1, 2026 + (day-1)
  return date.toISOString().split('T')[0];
}

// Group lessons by day
const byDay = {};
lessons.forEach(l => {
  if (!byDay[l.day]) byDay[l.day] = {};
  byDay[l.day][l.track] = l;
});

// Build structured output
const output = {
  version: "1.0.0",
  generated: new Date().toISOString().split('T')[0],
  description: "True North - Complete 365-Day Curriculum for Curious Kelly (2026 Calendar Year)",
  calendar_year: 2026,
  total_lessons: 365,
  tracks: ["learn", "grow"],
  source: "Supabase core_lessons table",
  lessons: []
};

for (let day = 1; day <= 365; day++) {
  const data = byDay[day] || {};
  const learn = data.learn || null;
  const grow = data.grow || null;
  
  output.lessons.push({
    day: day,
    date: getDate2026(day),
    learn: learn ? {
      id: learn.id,
      topic: learn.topic,
      headline: learn.headline || '',
      truth: learn.truth || '',
      icon: learn.icon || '📚',
      objectives: learn.objectives || [],
      difficulty: learn.difficulty || 'Beginner',
      duration_minutes: learn.duration || 8
    } : null,
    grow: grow ? {
      id: grow.id,
      topic: grow.topic,
      headline: grow.headline || '',
      truth: grow.truth || '',
      icon: grow.icon || '🌱'
    } : null
  });
}

// Write JSON
fs.writeFileSync('true-north.json', JSON.stringify(output, null, 2));
console.log('✅ Created true-north.json with', output.lessons.length, 'days');

// Generate HTML
let html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>True North - 365 Day Curriculum | Curious Kelly</title>
  <style>
    :root {
      --bg: #0f0f23;
      --card: #1a1a2e;
      --text: #e0e0e0;
      --accent: #00d4ff;
      --learn: #4ade80;
      --grow: #f472b6;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      padding: 2rem;
    }
    h1 {
      text-align: center;
      font-size: 2.5rem;
      margin-bottom: 0.5rem;
      background: linear-gradient(135deg, var(--learn), var(--accent), var(--grow));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .subtitle {
      text-align: center;
      color: #888;
      margin-bottom: 2rem;
    }
    .stats {
      display: flex;
      justify-content: center;
      gap: 2rem;
      margin-bottom: 2rem;
    }
    .stat {
      background: var(--card);
      padding: 1rem 2rem;
      border-radius: 12px;
      text-align: center;
    }
    .stat-value {
      font-size: 2rem;
      font-weight: bold;
      color: var(--accent);
    }
    .search {
      max-width: 600px;
      margin: 0 auto 2rem;
    }
    .search input {
      width: 100%;
      padding: 1rem;
      border: 2px solid #333;
      border-radius: 8px;
      background: var(--card);
      color: var(--text);
      font-size: 1rem;
    }
    .lessons {
      max-width: 1200px;
      margin: 0 auto;
    }
    .lesson {
      background: var(--card);
      border-radius: 12px;
      margin-bottom: 1rem;
      overflow: hidden;
    }
    .lesson-header {
      display: flex;
      align-items: center;
      padding: 1rem;
      cursor: pointer;
      gap: 1rem;
    }
    .lesson-header:hover {
      background: rgba(255,255,255,0.05);
    }
    .day-badge {
      background: var(--accent);
      color: #000;
      font-weight: bold;
      padding: 0.5rem 1rem;
      border-radius: 8px;
      min-width: 80px;
      text-align: center;
    }
    .date {
      color: #666;
      min-width: 100px;
    }
    .topics {
      flex: 1;
      display: flex;
      gap: 1rem;
    }
    .track-badge {
      padding: 0.25rem 0.75rem;
      border-radius: 20px;
      font-size: 0.9rem;
    }
    .track-learn { background: rgba(74,222,128,0.2); color: var(--learn); }
    .track-grow { background: rgba(244,114,182,0.2); color: var(--grow); }
    .lesson-content {
      display: none;
      padding: 0 1rem 1rem;
      border-top: 1px solid #333;
    }
    .lesson.open .lesson-content {
      display: block;
    }
    .track-section {
      margin-top: 1rem;
    }
    .track-title {
      font-weight: bold;
      margin-bottom: 0.5rem;
    }
    .headline {
      font-style: italic;
      color: #aaa;
      margin-bottom: 0.5rem;
    }
    .truth {
      color: var(--accent);
      margin-bottom: 0.5rem;
    }
    .meta {
      font-size: 0.85rem;
      color: #666;
    }
    .objectives {
      margin-top: 0.5rem;
      padding-left: 1.5rem;
    }
    .objectives li {
      margin-bottom: 0.25rem;
    }
  </style>
</head>
<body>
  <h1>✨ True North</h1>
  <p class="subtitle">The Complete 365-Day Curriculum for Curious Kelly - 2026</p>
  
  <div class="stats">
    <div class="stat">
      <div class="stat-value">365</div>
      <div>Days</div>
    </div>
    <div class="stat">
      <div class="stat-value">730</div>
      <div>Total Lessons</div>
    </div>
    <div class="stat">
      <div class="stat-value">2</div>
      <div>Tracks</div>
    </div>
  </div>
  
  <div class="search">
    <input type="text" id="searchInput" placeholder="Search lessons by topic, headline, or truth..." oninput="filterLessons()">
  </div>
  
  <div class="lessons" id="lessonsContainer">
`;

output.lessons.forEach(lesson => {
  const learnTopic = lesson.learn ? lesson.learn.topic : 'TBD';
  const growTopic = lesson.grow ? lesson.grow.topic : 'TBD';
  
  html += `
    <div class="lesson" data-search="${(learnTopic + ' ' + growTopic + ' ' + (lesson.learn?.headline || '') + ' ' + (lesson.grow?.headline || '') + ' ' + (lesson.learn?.truth || '') + ' ' + (lesson.grow?.truth || '')).toLowerCase()}">
      <div class="lesson-header" onclick="this.parentElement.classList.toggle('open')">
        <div class="day-badge">Day ${lesson.day}</div>
        <div class="date">${lesson.date}</div>
        <div class="topics">
          <span class="track-badge track-learn">${lesson.learn?.icon || '📚'} ${learnTopic}</span>
          <span class="track-badge track-grow">${lesson.grow?.icon || '🌱'} ${growTopic}</span>
        </div>
      </div>
      <div class="lesson-content">
        ${lesson.learn ? `
        <div class="track-section">
          <div class="track-title" style="color:var(--learn)">📚 LEARN Track: ${lesson.learn.topic}</div>
          <div class="headline">"${lesson.learn.headline}"</div>
          <div class="truth">💡 ${lesson.learn.truth}</div>
          <div class="meta">ID: ${lesson.learn.id} | ${lesson.learn.difficulty} | ${lesson.learn.duration_minutes} min</div>
          ${lesson.learn.objectives?.length ? `<ul class="objectives">${lesson.learn.objectives.map(o => '<li>' + o + '</li>').join('')}</ul>` : ''}
        </div>` : ''}
        ${lesson.grow ? `
        <div class="track-section">
          <div class="track-title" style="color:var(--grow)">🌱 GROW Track: ${lesson.grow.topic}</div>
          <div class="headline">"${lesson.grow.headline}"</div>
          <div class="truth">💡 ${lesson.grow.truth}</div>
          <div class="meta">ID: ${lesson.grow.id}</div>
        </div>` : ''}
      </div>
    </div>
`;
});

html += `
  </div>
  
  <script>
    function filterLessons() {
      const query = document.getElementById('searchInput').value.toLowerCase();
      document.querySelectorAll('.lesson').forEach(el => {
        const matches = el.dataset.search.includes(query);
        el.style.display = matches ? 'block' : 'none';
      });
    }
  </script>
</body>
</html>`;

fs.writeFileSync('true-north.html', html);
console.log('✅ Created true-north.html');
