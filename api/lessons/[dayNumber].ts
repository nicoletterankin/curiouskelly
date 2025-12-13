/**
 * Lessons API - Dynamic Day Number Endpoint
 * 
 * GET /api/lessons/:dayNumber
 * 
 * Serves lessons as a fallback when Supabase and D1 are unavailable.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

// Extended static lesson data (first 30 days)
const STATIC_LESSONS: Record<number, { topic: string; universal_truth: string; greeting: string }> = {
  1: { topic: 'The Sun', universal_truth: 'Our star gives life to everything on Earth.', greeting: 'Let\'s explore the incredible power of the Sun!' },
  2: { topic: 'Why the Sky is Blue', universal_truth: 'Light bends and scatters to paint our world.', greeting: 'Have you ever wondered why the sky is blue?' },
  3: { topic: 'How Seeds Grow', universal_truth: 'Every giant oak began as a tiny seed with potential.', greeting: 'Today we\'re planting seeds of knowledge!' },
  4: { topic: 'The Water Cycle', universal_truth: 'Water is never created or destroyed—only transformed.', greeting: 'The water you drink might have been drunk by a dinosaur!' },
  5: { topic: 'Why We Sleep', universal_truth: 'Sleep is when your brain organizes everything you learned.', greeting: 'Ready to learn why sleep is your superpower?' },
  6: { topic: 'How Birds Fly', universal_truth: 'Nature solved flight millions of years before humans.', greeting: 'Let\'s soar into the science of flight!' },
  7: { topic: 'The Moon', universal_truth: 'The Moon has watched over Earth for 4.5 billion years.', greeting: 'Tonight, look up at our closest neighbor in space!' },
  8: { topic: 'The Heart', universal_truth: 'Your heart never takes a break—it\'s always working for you.', greeting: 'Let\'s explore the most amazing pump in the world!' },
  9: { topic: 'The Brain', universal_truth: 'Your brain is the most complex object in the known universe.', greeting: 'Ready to explore the most amazing organ in your body?' },
  10: { topic: 'How We See', universal_truth: 'Your eyes capture light, but your brain creates the picture.', greeting: 'Let\'s see how seeing really works!' },
  11: { topic: 'Rainbows', universal_truth: 'White light contains all the colors of the rainbow.', greeting: 'Let\'s chase rainbows together!' },
  12: { topic: 'Volcanoes', universal_truth: 'Volcanoes remind us that Earth is still alive and changing.', greeting: 'Get ready for an explosive lesson!' },
  13: { topic: 'Dinosaurs', universal_truth: 'Dinosaurs ruled Earth for 165 million years.', greeting: 'Let\'s travel back in time 66 million years!' },
  14: { topic: 'The Ocean', universal_truth: 'The ocean holds 97% of all water on Earth.', greeting: 'Dive deep with me into the ocean\'s mysteries!' },
  15: { topic: 'How Computers Think', universal_truth: 'Everything a computer does comes down to 1s and 0s.', greeting: 'Let\'s peek inside the digital brain!' },
};

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Cache-Control', 'public, max-age=300');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  // Extract day number from URL
  const { dayNumber, archetype, ageBucket } = req.query;
  const day = parseInt(Array.isArray(dayNumber) ? dayNumber[0] : dayNumber || '1');
  
  if (isNaN(day) || day < 1 || day > 365) {
    return res.status(400).json({ 
      error: 'Invalid day number',
      message: 'Day must be between 1 and 365'
    });
  }
  
  // Get lesson from static data (cycle through available lessons)
  const lessonKey = day <= 15 ? day : ((day - 1) % 15) + 1;
  const lessonData = STATIC_LESSONS[lessonKey];
  
  const lesson = {
    id: `api-${day}`,
    day_number: day,
    topic: lessonData.topic,
    universal_truth: lessonData.universal_truth,
    marketing_headline: lessonData.topic,
    marketing_tagline: 'Discover something amazing today'
  };
  
  const atoms = [
    { phase: 'Hook', content: { script: lessonData.greeting, text: lessonData.greeting } },
    { phase: 'Fact1', content: { script: `Did you know? ${lessonData.universal_truth}`, text: lessonData.universal_truth } },
    { phase: 'Fact2', content: { script: `Here\'s something fascinating about ${lessonData.topic}...`, text: 'More fascinating facts await!' } },
    { phase: 'Fact3', content: { script: `And here\'s one more amazing thing about ${lessonData.topic}...`, text: 'One more amazing discovery!' } },
    { phase: 'Wisdom', content: { script: `Remember: ${lessonData.universal_truth}`, text: lessonData.universal_truth } }
  ];
  
  return res.status(200).json({
    source: 'api-static',
    lesson,
    atoms,
    shards: [],
    dayNumber: day,
    archetype: archetype || 'The Scientist',
    ageBucket: ageBucket || 'adult'
  });
}
