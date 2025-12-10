/**
 * Dynamic OG Image Generator
 * 
 * Generates social sharing images for each lesson.
 * /og/day-{number}.png
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

/**
 * Generate SVG-based OG image
 * Returns an SVG that can be converted to PNG
 */
function generateOGSVG(emoji: string, title: string, dayNumber: number): string {
  // Escape HTML entities in title
  const escapedTitle = title
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

  return `
<svg width="1200" height="630" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#fafaf9"/>
      <stop offset="100%" style="stop-color:#f5f5f4"/>
    </linearGradient>
  </defs>
  
  <!-- Background -->
  <rect width="1200" height="630" fill="url(#bg)"/>
  
  <!-- Subtle border -->
  <rect x="40" y="40" width="1120" height="550" rx="16" fill="white" stroke="#e7e5e4" stroke-width="1"/>
  
  <!-- Day badge -->
  <text x="100" y="120" font-family="system-ui, sans-serif" font-size="18" font-weight="600" fill="#78716c" letter-spacing="0.1em">
    DAY ${dayNumber} OF 365
  </text>
  
  <!-- Emoji -->
  <text x="100" y="260" font-size="120">${emoji}</text>
  
  <!-- Title -->
  <text x="100" y="380" font-family="Georgia, serif" font-size="56" font-weight="500" fill="#1c1917">
    ${escapedTitle.length > 28 ? escapedTitle.substring(0, 28) + '...' : escapedTitle}
  </text>
  
  <!-- Branding -->
  <text x="100" y="520" font-family="system-ui, sans-serif" font-size="24" fill="#a8a29e">
    ✨ curiouskelly.com
  </text>
</svg>
  `.trim();
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const { day } = req.query;
  
  // Parse day number from "day-123.png" format
  const dayMatch = (day as string).match(/day-(\d+)/);
  if (!dayMatch) {
    return res.status(400).json({ error: 'Invalid format. Use /og/day-{number}.png' });
  }
  
  const dayNumber = parseInt(dayMatch[1], 10);
  
  if (isNaN(dayNumber) || dayNumber < 1 || dayNumber > 365) {
    return res.status(404).json({ error: 'Invalid day number' });
  }

  if (!supabaseUrl || !supabaseServiceKey) {
    return res.status(500).json({ error: 'Server configuration error' });
  }

  const supabase = createClient(supabaseUrl, supabaseServiceKey);

  try {
    const { data: lesson, error } = await supabase
      .from('lessons')
      .select('title, emoji')
      .eq('day_number', dayNumber)
      .single();

    if (error || !lesson) {
      // Return generic image for missing lessons
      const svg = generateOGSVG('📚', `Day ${dayNumber}`, dayNumber);
      res.setHeader('Content-Type', 'image/svg+xml');
      res.setHeader('Cache-Control', 's-maxage=86400, stale-while-revalidate');
      return res.status(200).send(svg);
    }

    const svg = generateOGSVG(lesson.emoji || '📚', lesson.title, dayNumber);
    
    res.setHeader('Content-Type', 'image/svg+xml');
    res.setHeader('Cache-Control', 's-maxage=86400, stale-while-revalidate');
    
    return res.status(200).send(svg);

  } catch (error) {
    console.error('OG image error:', error);
    return res.status(500).json({ error: 'Something went wrong' });
  }
}



