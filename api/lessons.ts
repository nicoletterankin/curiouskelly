/**
 * Lessons API - Vercel Edge Function
 * 
 * Serves lessons from static JSON files as a fallback layer.
 * This runs on Vercel Edge and provides a simple API for lesson retrieval.
 * 
 * GET /api/lessons/:dayNumber
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

// Simple static lesson data for fallback
const STATIC_LESSONS: Record<number, any> = {
  1: { topic: 'The Sun', universal_truth: 'Our star gives life to everything on Earth.' },
  2: { topic: 'Why the Sky is Blue', universal_truth: 'Light bends and scatters to paint our world.' },
  3: { topic: 'How Seeds Grow', universal_truth: 'Every giant oak began as a tiny seed.' },
  4: { topic: 'The Water Cycle', universal_truth: 'Water is never created or destroyed—only transformed.' },
  5: { topic: 'Why We Sleep', universal_truth: 'Sleep is when your brain organizes everything you learned.' },
  6: { topic: 'How Birds Fly', universal_truth: 'Nature solved flight millions of years before humans.' },
  7: { topic: 'The Moon', universal_truth: 'The Moon has watched over Earth for 4.5 billion years.' },
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
  const { dayNumber } = req.query;
  const day = parseInt(Array.isArray(dayNumber) ? dayNumber[0] : dayNumber || '1');
  
  if (isNaN(day) || day < 1 || day > 365) {
    return res.status(400).json({ 
      error: 'Invalid day number',
      message: 'Day must be between 1 and 365'
    });
  }
  
  // Try to serve from static data
  const staticLesson = STATIC_LESSONS[day] || STATIC_LESSONS[(day % 7) + 1];
  
  if (staticLesson) {
    return res.status(200).json({
      source: 'api-static',
      lesson: {
        id: `api-${day}`,
        day_number: day,
        topic: staticLesson.topic,
        universal_truth: staticLesson.universal_truth,
        marketing_headline: staticLesson.topic
      },
      atoms: [
        { phase: 'Hook', content: { script: `Today we're learning about ${staticLesson.topic}!` } },
        { phase: 'Fact1', content: { script: staticLesson.universal_truth } },
        { phase: 'Fact2', content: { script: 'Here\'s something interesting about this topic...' } },
        { phase: 'Fact3', content: { script: 'And one more fascinating fact...' } },
        { phase: 'Wisdom', content: { script: staticLesson.universal_truth } }
      ],
      shards: [],
      dayNumber: day
    });
  }
  
  // Fallback for any day
  return res.status(200).json({
    source: 'api-fallback',
    lesson: {
      id: `fallback-${day}`,
      day_number: day,
      topic: 'Daily Discovery',
      universal_truth: 'Every day brings something new to learn.',
      marketing_headline: 'Discover something amazing'
    },
    atoms: [
      { phase: 'Hook', content: { script: 'Welcome to today\'s learning adventure!' } },
      { phase: 'Fact1', content: { script: 'Let\'s explore together.' } },
      { phase: 'Fact2', content: { script: 'Here\'s something interesting...' } },
      { phase: 'Fact3', content: { script: 'And one more thing...' } },
      { phase: 'Wisdom', content: { script: 'Knowledge is power.' } }
    ],
    shards: [],
    dayNumber: day
  });
}

export const config = {
  runtime: 'edge',
};
