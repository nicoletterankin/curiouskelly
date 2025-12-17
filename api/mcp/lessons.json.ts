/**
 * MCP Lessons JSON Endpoint
 * 
 * Machine-readable lesson data for AI agents (MCPs)
 * Returns JSON-LD format with all 365 lessons
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin, isSupabaseConfigured } from '../lib/supabase';

const BASE_URL = 'https://curiouskelly.com';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  let lessons: any[] = [];
  
  if (isSupabaseConfigured()) {
    try {
      const supabase = getSupabaseAdmin();
      
      const { data } = await supabase
        .from('core_lessons')
        .select('day_number, topic, universal_truth, marketing_headline, marketing_tagline')
        .order('day_number');
      
      if (data) {
        lessons = data;
      }
    } catch (e) {
      console.warn('Failed to fetch lessons for MCP:', e);
    }
  }
  
  const response = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    "name": "Curious Kelly Daily Lessons",
    "description": "365 daily learning experiences for ages 2-102. One topic, 12 teachers, personalized for you.",
    "url": BASE_URL,
    "numberOfItems": lessons.length || 365,
    "publisher": {
      "@type": "Organization",
      "name": "Lesson of the Day PBC",
      "url": BASE_URL
    },
    "itemListElement": lessons.map((lesson, i) => ({
      "@type": "ListItem",
      "position": i + 1,
      "item": {
        "@type": "LearningResource",
        "identifier": `day-${lesson.day_number}`,
        "name": lesson.topic,
        "description": lesson.marketing_headline || lesson.universal_truth,
        "tagline": lesson.marketing_tagline,
        "educationalLevel": "All ages (2-102)",
        "timeRequired": "PT5M",
        "url": `${BASE_URL}/day/${lesson.day_number}`,
        "teaches": lesson.universal_truth
      }
    })),
    "meta": {
      "totalLessons": lessons.length || 365,
      "totalAtoms": 21915,
      "totalShards": 38700,
      "archetypes": [
        "The Scientist",
        "The Explorer", 
        "The Rebel",
        "The Architect",
        "The Diplomat",
        "The Empath",
        "The MacGyver",
        "The Mystic",
        "The Provider",
        "The Storyteller",
        "The Strategist",
        "The Survivor"
      ],
      "ageRegions": ["kid", "teen", "adult", "mature", "elder"],
      "apiVersion": "1.0.0",
      "generatedAt": new Date().toISOString()
    }
  };
  
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Cache-Control', 'public, max-age=0, s-maxage=3600, stale-while-revalidate=86400');
  
  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }
  
  return res.status(200).json(response);
}







