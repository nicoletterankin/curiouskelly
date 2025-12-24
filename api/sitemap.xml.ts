/**
 * Dynamic Sitemap Generator
 * 
 * Generates sitemap.xml for SEO with all 365 lesson pages
 * Queries core_lessons table for proper content
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin, isSupabaseConfigured } from './lib/supabase';

const BASE_URL = 'https://curiouskelly.com';

// All 12 Kelly archetypes
const KELLYS = [
  'scientist', 'explorer', 'rebel', 'architect', 
  'diplomat', 'empath', 'macgyver', 'mystic',
  'provider', 'storyteller', 'strategist', 'survivor'
];

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const today = new Date().toISOString().split('T')[0];
  
  let lessonUrls = '';
  
  if (isSupabaseConfigured()) {
    try {
      const supabase = getSupabaseAdmin();
      
      // Fetch all lessons
      const { data: lessons } = await supabase
        .from('core_lessons')
        .select('day_number, topic')
        .order('day_number');
      
      if (lessons && lessons.length > 0) {
        lessonUrls = lessons.map(lesson => `
  <url>
    <loc>${BASE_URL}/day/${lesson.day_number}</loc>
    <lastmod>${today}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.7</priority>
  </url>`).join('');
      }
    } catch (e) {
      console.warn('Failed to fetch lessons for sitemap:', e);
    }
  }
  
  // If no lessons from DB, generate all 365
  if (!lessonUrls) {
    lessonUrls = Array.from({ length: 365 }, (_, i) => `
  <url>
    <loc>${BASE_URL}/day/${i + 1}</loc>
    <lastmod>${today}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.7</priority>
  </url>`).join('');
  }
  
  // Kelly profile pages
  const kellyUrls = KELLYS.map(kelly => `
  <url>
    <loc>${BASE_URL}/kelly/${kelly}</loc>
    <lastmod>${today}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.8</priority>
  </url>`).join('');
  
  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>${BASE_URL}/</loc>
    <lastmod>${today}</lastmod>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>${BASE_URL}/learn.html</loc>
    <lastmod>${today}</lastmod>
    <changefreq>daily</changefreq>
    <priority>0.9</priority>
  </url>
  <url>
    <loc>${BASE_URL}/calendar.html</loc>
    <lastmod>${today}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>
  <url>
    <loc>${BASE_URL}/curriculum.html</loc>
    <lastmod>${today}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>${kellyUrls}${lessonUrls}
</urlset>`;

  res.setHeader('Content-Type', 'application/xml');
  res.setHeader('Cache-Control', 'public, max-age=0, s-maxage=86400, stale-while-revalidate=604800');
  
  return res.status(200).send(xml);
}
















