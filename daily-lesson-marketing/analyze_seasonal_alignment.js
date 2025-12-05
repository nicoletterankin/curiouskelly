/**
 * Analyze topic-to-day alignment for seasonal appropriateness
 */

import fs from 'fs';

const data = JSON.parse(fs.readFileSync('enhanced_topics.json', 'utf8'));

// Month boundaries
const months = [
  { name: 'January', start: 1, end: 31 },
  { name: 'February', start: 32, end: 59 },
  { name: 'March', start: 60, end: 90 },
  { name: 'April', start: 91, end: 120 },
  { name: 'May', start: 121, end: 151 },
  { name: 'June', start: 152, end: 181 },
  { name: 'July', start: 182, end: 212 },
  { name: 'August', start: 213, end: 243 },
  { name: 'September', start: 244, end: 273 },
  { name: 'October', start: 274, end: 304 },
  { name: 'November', start: 305, end: 334 },
  { name: 'December', start: 335, end: 366 }
];

function getMonth(day) {
  return months.find(m => day >= m.start && day <= m.end)?.name || 'Unknown';
}

function getSeason(day) {
  if (day <= 59 || day >= 335) return 'Winter';
  if (day <= 151) return 'Spring';
  if (day <= 243) return 'Summer';
  return 'Fall';
}

console.log('╔═══════════════════════════════════════════════════════════╗');
console.log('║        📅 TOPIC-TO-DAY SEASONAL ANALYSIS                  ║');
console.log('╚═══════════════════════════════════════════════════════════╝\n');

// Show first two months
console.log('=== JANUARY (Days 1-31) ===\n');
data.topics.filter(t => t.day <= 31).forEach(t => {
  console.log(`  Day ${t.day.toString().padStart(2)}: ${t.new_topic}`);
});

console.log('\n=== FEBRUARY (Days 32-59) ===\n');
data.topics.filter(t => t.day >= 32 && t.day <= 59).forEach(t => {
  console.log(`  Day ${t.day.toString().padStart(2)}: ${t.new_topic}`);
});

// Topics that are clearly seasonal
const seasonalKeywords = {
  'Winter': ['ice', 'snow', 'cold', 'frost', 'winter', 'christmas', 'holiday'],
  'Spring': ['seed', 'flower', 'bloom', 'grow', 'spring', 'rain', 'plant', 'birth'],
  'Summer': ['sun', 'beach', 'ocean', 'heat', 'summer', 'vacation', 'swim'],
  'Fall': ['leaf', 'leaves', 'harvest', 'fall', 'autumn', 'change', 'decay']
};

console.log('\n=== SEASONAL TOPIC PLACEMENT CHECK ===\n');

const misalignments = [];

data.topics.forEach(t => {
  const topicLower = t.new_topic.toLowerCase();
  const oldTopicLower = t.old_topic.toLowerCase();
  const currentSeason = getSeason(t.day);
  const currentMonth = getMonth(t.day);
  
  for (const [idealSeason, keywords] of Object.entries(seasonalKeywords)) {
    for (const keyword of keywords) {
      if (topicLower.includes(keyword) || oldTopicLower.includes(keyword)) {
        if (idealSeason !== currentSeason) {
          misalignments.push({
            day: t.day,
            month: currentMonth,
            currentSeason,
            idealSeason,
            topic: t.new_topic,
            oldTopic: t.old_topic,
            keyword
          });
        }
        break;
      }
    }
  }
});

if (misalignments.length > 0) {
  console.log('⚠️  POTENTIAL MISALIGNMENTS:\n');
  misalignments.forEach(m => {
    console.log(`  Day ${m.day} (${m.month}, ${m.currentSeason}):`);
    console.log(`    Topic: "${m.topic}"`);
    console.log(`    Keyword: "${m.keyword}" → Better in ${m.idealSeason}`);
    console.log('');
  });
} else {
  console.log('✅ No obvious seasonal misalignments found!');
}

// Check for special days
console.log('\n=== SPECIAL DAY OPPORTUNITIES ===\n');

const specialDays = [
  { day: 1, name: 'New Year\'s Day', suggest: 'New Beginnings, Goals, Change' },
  { day: 14, name: 'Valentine\'s Day (Feb 14)', suggest: 'Love, Friendship, Kindness' },
  { day: 45, name: 'Valentine\'s Day (Feb 14)', suggest: 'Love, Heart, Connection' },
  { day: 79, name: 'Pi Day (Mar 14)', suggest: 'Math, Circles, Numbers' },
  { day: 81, name: 'St. Patrick\'s Day (Mar 17)', suggest: 'Luck, Culture, Traditions' },
  { day: 111, name: 'Earth Day (Apr 22)', suggest: 'Environment, Conservation' },
  { day: 152, name: 'Summer Solstice (~Jun 21)', suggest: 'Sun, Light, Seasons' },
  { day: 185, name: 'Independence Day (Jul 4)', suggest: 'Freedom, Democracy, Rights' },
  { day: 266, name: 'Autumn Equinox (~Sep 22)', suggest: 'Balance, Seasons, Change' },
  { day: 304, name: 'Halloween (Oct 31)', suggest: 'Fear, Imagination, Traditions' },
  { day: 329, name: 'Thanksgiving (~Nov 25)', suggest: 'Gratitude, Family' },
  { day: 355, name: 'Winter Solstice (~Dec 21)', suggest: 'Light, Darkness, Seasons' },
  { day: 359, name: 'Christmas (Dec 25)', suggest: 'Giving, Traditions, Family' },
  { day: 365, name: 'New Year\'s Eve', suggest: 'Reflection, Goals, Journey' }
];

specialDays.forEach(s => {
  const topic = data.topics.find(t => t.day === s.day);
  if (topic) {
    console.log(`  Day ${s.day} (${s.name}):`);
    console.log(`    Current: "${topic.new_topic}"`);
    console.log(`    Suggested themes: ${s.suggest}`);
    console.log('');
  }
});

// Summary
console.log('\n=== RECOMMENDATION ===\n');
console.log('Key fixes needed:');
console.log('  • Day 1 (Jan 1): "How Leaves Feed the World" → Should be about NEW BEGINNINGS');
console.log('  • Day 365 (Dec 31): Should be about REFLECTION or JOURNEY');
console.log('  • Consider seasonal topics for:');
console.log('    - Winter: Ice, Snow, Cold, Rest, Hibernation');
console.log('    - Spring: Seeds, Growth, Flowers, Rain, Birth');
console.log('    - Summer: Sun, Heat, Energy, Ocean, Light');
console.log('    - Fall: Leaves, Harvest, Change, Decay');








