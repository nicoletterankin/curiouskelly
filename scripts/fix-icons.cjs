const fs = require('fs');

// Read the Supabase export
const raw = fs.readFileSync('C:/Users/user/.cursor/projects/c-Users-user-UI-TARS-desktop/agent-tools/5401cfd2-713d-4c29-b9a5-98039e35a994.txt', 'utf8');
const match = raw.match(/\[.*\]/s);
if (!match) {
  console.log('No data found');
  process.exit(1);
}

const data = JSON.parse(match[0]);
console.log('Total LEARN lessons:', data.length);

// Topic to correct icon mapping (based on topic name logic)
const topicIcons = {
  'Starting Fresh': '🌅',
  'The Three Lives of Water': '💧',
  'Where Clouds Come From': '☁️',
  'How Light Travels': '💡',
  'How Sound Moves': '🔊',
  "What's Inside a Seed": '🌱',
  'What Stars Are Made Of': '⭐',
  'What Makes a Real Friend': '🤝',
  'How Kindness Spreads': '💝',
  'The Art of Really Listening': '👂',
  'Why Patience Pays Off': '⏳',
  'How Gratitude Changes You': '🙏',
  'What Courage Really Means': '🦁',
  'Why Curious People Learn More': '🔍',
  'How Your Body Stays Balanced': '⚖️',
  'Why Breathing Matters': '🌬️',
  'Why Bodies Need to Move': '🏃',
  'What Happens When You Rest': '😴',
  'How Energy Changes Form': '⚡',
  'Your Five Senses (And More)': '👁️',
  'What Makes Things Grow': '🌿',
  'Why We See Colors': '🌈',
  'Patterns Are Everywhere': '🔷',
  'Why Humans Tell Stories': '📖',
  'Why Music Moves Us': '🎵',
  'The Power of Good Questions': '❓',
  'How Imagination Works': '💭',
  'How Memories Are Made': '🧠',
  'Why Time Feels Different': '⏰',
  'Why Everything Changes': '🔄',
  'How the Sun Powers Earth': '☀️',
  'The Moon and the Tides': '🌙',
  'What Gravity Actually Does': '🍎',
  'How Magnets Work': '🧲',
  'How Electricity Flows': '⚡',
  'What Fire Really Is': '🔥',
  'Why Ice Floats': '🧊',
  'What Makes Wind Blow': '💨',
  'Where Rain Comes From': '🌧️',
  'What Causes Thunder': '⛈️',
  'How Rainbows Form': '🌈',
  'Why Seasons Change': '🍂',
  'Why We Have Day and Night': '🌓',
  'How Shadows Work': '👥',
  'Why Mirrors Reflect': '🪞',
  'How Sound Bounces Back': '📣',
  'How Waves Carry Energy': '🌊',
  'The Science of Bubbles': '🫧',
  'How Crystals Form': '💎',
  'Stories Trapped in Stone': '🦴',
  'When Dinosaurs Ruled': '🦖',
  "What's Inside a Volcano": '🌋',
  'Why the Ground Shakes': '🌍',
  'How Mountains Are Made': '⛰️',
  'The Deep Ocean Mystery': '🌊',
  'How Rivers Shape the Land': '🏞️',
  'Where Lakes Come From': '🏔️',
  'Life in the Desert': '🌵',
  'The Secret Life of Forests': '🌲',
  'Why Jungles Are So Alive': '🌴',
  'The Power of Grass': '🌾',
  'Why Wetlands Matter': '🐸',
  'Cities Under the Sea': '🪸',
  'Worlds Without Light': '🦇',
  'How Islands Are Born': '🏝️',
  "What's Living in the Dirt": '🪱',
  'The Stories Rocks Tell': '🪨',
  "Earth's Hidden Treasures": '💎',
  'How Gems Are Made': '💍',
  'Where Metals Come From': '⚙️',
  "What's In the Air You Breathe": '🌬️',
  'Why We Need Oxygen': '🫁',
  'Carbon Is Everywhere': '⚫',
  "The Gas You Don't Notice": '🌫️',
  'The Simplest Element': '⚛️',
  'Building Blocks of Everything': '🧱',
  'When Atoms Connect': '🔗',
  'The Tiny Units of Life': '🔬',
  "Your Body's Instruction Manual": '🧬',
  'What Blood Does All Day': '🩸',
  'How Taste Works': '👅',
  'Why Smell Triggers Memory': '👃',
  'How Skin Feels Things': '🖐️',
  'How Your Eyes See': '👁️',
  'How Your Ears Work': '👂',
  'Your Heart Never Stops': '❤️',
  'What Your Brain Does': '🧠',
  'How Lungs Breathe for You': '🫁',
  "Your Body's Framework": '🦴',
  'What Love Does to Us': '💕',
  'What Makes a Family': '👨‍👩‍👧',
  'How We Understand Each Other': '🗣️',
  'Why Humans Have Language': '💬',
  'When Humans Started Writing': '✍️',
  'What Reading Does to Your Brain': '📚',
  'Why Numbers Were Invented': '🔢',
  'How Adding Works': '➕',
  'Taking Things Away': '➖',
  'The Shortcut for Repeated Adding': '✖️',
  'Splitting Things Fairly': '➗'
};

// Find mismatches
let mismatches = [];
let sqlStatements = [];

data.forEach(lesson => {
  const correctIcon = topicIcons[lesson.topic];
  if (correctIcon && correctIcon !== lesson.icon_emoji) {
    mismatches.push({
      day: lesson.day_number,
      topic: lesson.topic,
      current: lesson.icon_emoji,
      correct: correctIcon
    });
    sqlStatements.push(
      `UPDATE core_lessons SET icon_emoji = '${correctIcon}' WHERE track = 'learn' AND day_number = ${lesson.day_number};`
    );
  }
});

console.log('\nIcon mismatches found:', mismatches.length);
console.log('\n=== MISMATCHES ===');
mismatches.forEach(m => {
  console.log(`Day ${m.day}: ${m.topic} | ${m.current} -> ${m.correct}`);
});

console.log('\n=== SQL FIX STATEMENTS ===');
sqlStatements.forEach(sql => console.log(sql));

// Save to file
fs.writeFileSync('icon-fix-sql.txt', sqlStatements.join('\n'));
console.log('\nSQL saved to icon-fix-sql.txt');
