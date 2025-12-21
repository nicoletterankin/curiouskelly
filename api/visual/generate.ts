/**
 * Visual Commons - Generate Endpoint
 * 
 * Generates a new visual for the given context, caches it, and returns the URL.
 * 
 * POST /api/visual/generate
 * Body: { dayNumber, phase, ageGroup, visualType, userApiKey? }
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';
import { GoogleGenerativeAI } from '@google/generative-ai';
import crypto from 'crypto';

// Types
interface GenerateRequest {
  dayNumber: number;
  phase: string;
  ageGroup?: string;
  visualType?: string;
  style?: string;
  userApiKey?: string; // BYOK
}

interface VisualContext {
  dayNumber: number;
  phase: string;
  ageGroup: string;
  visualType: string;
  style: string;
  topic?: string;
  universalTruth?: string;
  facts?: string[];
}

// Initialize Supabase
const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || '';
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || '';

function getSupabase() {
  return createClient(supabaseUrl, supabaseKey);
}

// Generate content hash
function generateVisualHash(context: VisualContext): string {
  const normalized = {
    d: context.dayNumber,
    p: context.phase.toLowerCase(),
    a: context.ageGroup || 'all',
    t: context.visualType || 'infographic',
    s: context.style || 'default',
    ver: '1'
  };
  
  const canonical = JSON.stringify(normalized, Object.keys(normalized).sort());
  return crypto.createHash('sha256').update(canonical).digest('hex');
}

// Get lesson details from database
async function getLessonDetails(dayNumber: number): Promise<{ topic: string; universalTruth: string; facts: string[] } | null> {
  const supabase = getSupabase();
  const { data } = await supabase
    .from('core_lessons')
    .select('topic, universal_truth, facts')
    .eq('day_number', dayNumber)
    .eq('track', 'learn')  // Default to learn track
    .maybeSingle();
  
  if (!data) return null;
  
  return {
    topic: data.topic || `Day ${dayNumber} Lesson`,
    universalTruth: data.universal_truth || '',
    facts: data.facts || []
  };
}

// Build the generation prompt
function buildPrompt(context: VisualContext): string {
  const ageAdaptations: Record<string, string> = {
    '2-5': 'Use bright colors, friendly rounded shapes, and simple concepts for young children.',
    '6-12': 'Balance fun with accuracy, include cool elements, make them feel smart.',
    '13-17': 'Sophisticated visuals, can include complexity, avoid anything babyish.',
    '18+': 'Full scientific accuracy, professional aesthetic, respect intelligence.',
    'all': 'Universal visual language, clear at surface, richer for those who look closer.'
  };

  const phaseGoals: Record<string, string> = {
    hook: 'Create curiosity and intrigue. Make them say "Wait, what?!"',
    cliff: 'Deepen the mystery. Show the gap between assumption and truth.',
    fact1: 'Crystal clear teaching. One main idea, well explained.',
    fact2: 'Build on previous knowledge. Show connections and depth.',
    fact3: 'The wow moment. Surprising but true, memorable.',
    wisdom: 'Life application. Make it personal and poster-worthy.',
    outro: 'Celebrate and tease tomorrow. End on positive energy.',
    complete: 'Comprehensive summary. One image that teaches the whole lesson.'
  };

  return `You are Kelly's Visual Design Lead for Curious Kelly, an educational platform.

TASK: Generate a structured infographic brief as JSON.

BRAND:
- Dark premium backgrounds (#0a0a0b, #18181b)
- Neon accents (#3b82f6 blue, #fbbf24 gold, #22c55e green)
- Clean, minimal typography

CONSTRAINTS:
- Labels ≤4 words, details ≤18 words
- Scientifically accurate
- Age-appropriate for ${context.ageGroup}

AGE ADAPTATION: ${ageAdaptations[context.ageGroup] || ageAdaptations['all']}

PHASE: ${context.phase.toUpperCase()}
GOAL: ${phaseGoals[context.phase] || 'Create an educational visual.'}

TOPIC: ${context.topic || 'Educational Topic'}
UNIVERSAL TRUTH: ${context.universalTruth || 'Not specified'}

TEMPLATE OPTIONS:
1. cross_section - Layered diagram showing internal structure
2. process_flow - 3-step horizontal flow with arrows  
3. compare - Two-panel side-by-side comparison
4. radial - Central concept with orbital related ideas
5. timeline - Chronological sequence

OUTPUT SCHEMA (return ONLY this JSON, no explanation):
{
  "template": "cross_section" | "process_flow" | "compare" | "radial" | "timeline",
  "headline": "8 words max",
  "subhead": "16 words max",
  "callouts": [
    { "label": "4 words max", "detail": "18 words max", "icon": "atom|spark|arrow|leaf|heart|wave|dot|star|bulb" }
  ]
}`;
}

// Generate infographic brief using Gemini
async function generateInfographicBrief(
  context: VisualContext, 
  apiKey: string
): Promise<{ brief: any; generationTimeMs: number }> {
  const startTime = Date.now();
  
  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });
  
  const prompt = buildPrompt(context);
  
  const result = await model.generateContent({
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    generationConfig: { 
      responseMimeType: 'application/json',
      temperature: 0.7
    }
  });
  
  const responseText = result.response.text();
  const brief = JSON.parse(responseText);
  
  return {
    brief,
    generationTimeMs: Date.now() - startTime
  };
}

// Render SVG from brief (simplified - full version in visual-prompts.ts)
function renderSvgFromBrief(brief: any): string {
  const BRAND = {
    bg: '#0a0a0b',
    card: '#18181b',
    border: '#27272a',
    text: '#f4f4f5',
    muted: '#a1a1aa',
    accent: '#3b82f6',
    gold: '#fbbf24'
  };

  const W = 1344;
  const H = 768;
  const pad = 48;

  // Escape XML entities
  const esc = (s: string) => s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

  const callouts = (brief.callouts || []).slice(0, 5);
  const calloutRows = callouts.map((c: any, i: number) => {
    const y = 200 + i * 100;
    return `
      <g>
        <circle cx="${pad + 20}" cy="${y}" r="8" fill="${BRAND.accent}"/>
        <text x="${pad + 40}" y="${y - 6}" fill="${BRAND.text}" font-size="18" font-weight="700" font-family="system-ui">${esc(c.label || '')}</text>
        <text x="${pad + 40}" y="${y + 18}" fill="${BRAND.muted}" font-size="14" font-family="system-ui">${esc(c.detail || '')}</text>
      </g>`;
  }).join('\n');

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="${BRAND.bg}"/>
      <stop offset="1" stop-color="#101013"/>
    </linearGradient>
  </defs>
  
  <rect width="${W}" height="${H}" fill="url(#bg)"/>
  
  <text x="${pad}" y="${pad + 40}" fill="${BRAND.text}" font-size="36" font-weight="800" font-family="system-ui">${esc(brief.headline || 'Visual Learning')}</text>
  <text x="${pad}" y="${pad + 80}" fill="${BRAND.muted}" font-size="18" font-family="system-ui">${esc(brief.subhead || '')}</text>
  
  <rect x="${pad}" y="140" width="${W - pad * 2}" height="${H - 180}" rx="16" fill="${BRAND.card}" stroke="${BRAND.border}"/>
  
  ${calloutRows}
  
  <g opacity="0.6">
    <text x="${W - pad}" y="${H - pad}" text-anchor="end" fill="${BRAND.muted}" font-size="12" font-family="system-ui">✨ Curious Kelly • Visual Commons</text>
  </g>
</svg>`;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const body = req.body as GenerateRequest;
    
    // Validate required fields
    if (!body.dayNumber || !body.phase) {
      return res.status(400).json({ 
        error: 'Missing required fields',
        required: ['dayNumber', 'phase']
      });
    }

    const dayNumber = body.dayNumber;
    const phase = body.phase.toLowerCase();
    const ageGroup = body.ageGroup || 'all';
    const visualType = body.visualType || 'infographic';
    const style = body.style || 'default';

    // Get lesson details
    const lesson = await getLessonDetails(dayNumber);
    
    // Build context
    const context: VisualContext = {
      dayNumber,
      phase,
      ageGroup,
      visualType,
      style,
      topic: lesson?.topic || `Day ${dayNumber} Lesson`,
      universalTruth: lesson?.universalTruth,
      facts: lesson?.facts
    };

    // Generate hash
    const contentHash = generateVisualHash(context);

    // Check if already exists (race condition prevention)
    const supabase = getSupabase();
    const { data: existing } = await supabase
      .from('visual_commons')
      .select('id, public_url')
      .eq('content_hash', contentHash)
      .maybeSingle();

    if (existing) {
      return res.status(200).json({
        success: true,
        cached: true,
        visual: {
          id: existing.id,
          publicUrl: existing.public_url,
          contentHash
        }
      });
    }

    // Determine API key source
    const apiKey = body.userApiKey || process.env.GEMINI_API_KEY || process.env.GOOGLE_AI_API_KEY;
    const keySource = body.userApiKey ? 'byok' : 'platform';

    if (!apiKey) {
      return res.status(500).json({ error: 'No API key available' });
    }

    // Generate the infographic brief
    const { brief, generationTimeMs } = await generateInfographicBrief(context, apiKey);

    // Render SVG
    const svgContent = renderSvgFromBrief(brief);
    const svgBuffer = Buffer.from(svgContent, 'utf-8');

    // Upload to Supabase Storage
    const storagePath = `visuals/${contentHash}.svg`;
    const { error: uploadError } = await supabase.storage
      .from('visuals')
      .upload(storagePath, svgBuffer, {
        contentType: 'image/svg+xml',
        upsert: true
      });

    if (uploadError) {
      console.error('Upload error:', uploadError);
      return res.status(500).json({ error: 'Failed to upload visual', details: uploadError.message });
    }

    // Get public URL
    const { data: urlData } = supabase.storage
      .from('visuals')
      .getPublicUrl(storagePath);

    const publicUrl = urlData.publicUrl;

    // Get user info if authenticated
    let generatedBy = null;
    let displayName = 'A curious learner';
    
    // In production, extract user from auth header
    // For now, allow anonymous generation

    // Insert into database
    const { data: inserted, error: insertError } = await supabase
      .from('visual_commons')
      .insert({
        content_hash: contentHash,
        day_number: dayNumber,
        phase,
        topic: context.topic,
        visual_type: visualType,
        age_group: ageGroup,
        style,
        storage_path: storagePath,
        public_url: publicUrl,
        format: 'svg',
        prompt_used: JSON.stringify(brief),
        model_used: 'gemini-2.0-flash',
        generation_params: { temperature: 0.7 },
        generation_time_ms: generationTimeMs,
        estimated_cost: 0, // Gemini text is free
        generated_by: generatedBy,
        generated_by_display_name: displayName,
        generation_source: keySource,
        status: 'active'
      })
      .select('id')
      .single();

    if (insertError) {
      console.error('Insert error:', insertError);
      // Visual was uploaded but not registered - still return success
    }

    return res.status(200).json({
      success: true,
      cached: false,
      visual: {
        id: inserted?.id || contentHash,
        publicUrl,
        contentHash,
        generationTimeMs,
        keySource
      },
      attribution: {
        message: 'You just illuminated this lesson!',
        isFirstContributor: true
      }
    });

  } catch (error: any) {
    console.error('Visual generation error:', error);
    return res.status(500).json({ 
      error: 'Generation failed',
      message: error.message 
    });
  }
}
