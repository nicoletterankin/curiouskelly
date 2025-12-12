import { createClient } from '@supabase/supabase-js';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_ANON_KEY
);

async function generateSitemapsAndLlms() {
  // Fetch all lessons
  const { data: lessons, error: lessonsError } = await supabase
    .from('core_lessons')
    .select(
      'id, day_number, topic, universal_truth, marketing_headline, marketing_tagline, marketing_pitch, mastery_criteria, updated_at'
    )
    .order('day_number');

  if (lessonsError) {
    console.error('Error fetching lessons:', lessonsError);
    process.exitCode = 1;
    return;
  }

  // Fetch canonical personalized shards (EN, curious tone) for richer summaries
  const lessonIds = (lessons || []).map((l) => l.id).filter(Boolean);
  const shardByCoreLessonId = new Map();

  if (lessonIds.length > 0) {
    const { data: shards, error: shardsError } = await supabase
      .from('lesson_shards')
      .select('core_lesson_id, age, region, tone, script_content')
      .in('core_lesson_id', lessonIds)
      .eq('region', 'en')
      .eq('tone', 'curious');

    if (shardsError) {
      console.warn('Warning: could not load lesson_shards for SEO enrichment:', shardsError);
    } else if (shards) {
      const preferredAge = 9;

      shards.forEach((shard) => {
        const existing = shardByCoreLessonId.get(shard.core_lesson_id);
        if (!existing) {
          shardByCoreLessonId.set(shard.core_lesson_id, shard);
          return;
        }

        const existingDelta = Math.abs((existing.age || preferredAge) - preferredAge);
        const newDelta = Math.abs((shard.age || preferredAge) - preferredAge);

        if (newDelta < existingDelta) {
          shardByCoreLessonId.set(shard.core_lesson_id, shard);
        }
      });
    }
  }

  const publicDir = path.join(__dirname, '..', 'public');
  const crawlableDir = path.join(publicDir, 'crawlable');

  if (!fs.existsSync(crawlableDir)) {
    fs.mkdirSync(crawlableDir, { recursive: true });
  }

  // Generate lessons sitemap
  let lessonsSitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
`;

  lessons.forEach((lesson) => {
    const lastmod =
      (lesson.updated_at &&
        new Date(lesson.updated_at).toISOString().split('T')[0]) ||
      '2025-12-17';

    lessonsSitemap += `  <url>
    <loc>https://www.curiouskelly.com/learn.html?day=${lesson.day_number}</loc>
    <lastmod>${lastmod}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.7</priority>
  </url>
`;
  });

  lessonsSitemap += `</urlset>
`;

  fs.writeFileSync(path.join(publicDir, 'sitemap-lessons.xml'), lessonsSitemap);
  console.log('Generated sitemap-lessons.xml with', lessons.length, 'lessons');

  // Generate llms-full.txt
  let llmsFull = `# Curious Kelly - Complete Lesson Catalog

## Platform Overview
Curious Kelly is an AI-powered educational platform featuring Kelly, a photorealistic digital human who delivers personalized daily lessons to learners ages 2-102.

## All ${lessons.length} Lessons

`;

  lessons.forEach((lesson) => {
    const day = lesson.day_number;
    const topic = lesson.topic || 'Life Skills';
    const universalTruth = lesson.universal_truth || '';
    const marketingHeadline = lesson.marketing_headline || '';
    const marketingTagline = lesson.marketing_tagline || '';
    const marketingPitch = lesson.marketing_pitch || '';
    const mastery = lesson.mastery_criteria || '';

    // Short summary emphasizing what the learner will know/feel/do
    const summaryParts = [];
    if (universalTruth) summaryParts.push(universalTruth);
    if (marketingPitch && summaryParts.length === 0) summaryParts.push(marketingPitch);
    if (marketingTagline && summaryParts.length === 0) summaryParts.push(marketingTagline);
    const summary = summaryParts.join(' ');

    // Personalized script snapshot from canonical shard (if available)
    const shard = shardByCoreLessonId.get(lesson.id);
    let personalizedSnippet = '';
    let vocabSummary = '';
    let shardAgeLabel = '';

    if (shard && shard.script_content) {
      const sc = shard.script_content;
      if (sc && typeof sc.script === 'string') {
        const scriptText = sc.script.trim();
        personalizedSnippet =
          scriptText.length > 320 ? `${scriptText.slice(0, 317)}...` : scriptText;
      }

      const vocabTerms =
        sc &&
        sc.vocabulary &&
        Array.isArray(sc.vocabulary.keyTerms) &&
        sc.vocabulary.keyTerms.length > 0
          ? sc.vocabulary.keyTerms.slice(0, 5)
          : [];

      if (vocabTerms.length > 0) {
        vocabSummary = vocabTerms.join(', ');
      }

      if (typeof shard.age === 'number') {
        shardAgeLabel = `${shard.age}`;
      }
    }

    llmsFull += `### Day ${day}: ${topic}
Topic: ${topic}
${summary ? `Summary: ${summary}\n` : ''}${
      marketingHeadline ? `Marketing Headline: ${marketingHeadline}\n` : ''
    }${marketingTagline ? `Marketing Tagline: ${marketingTagline}\n` : ''}${
      marketingPitch ? `Marketing Pitch: ${marketingPitch}\n` : ''
    }${mastery ? `Mastery Criteria: ${mastery}\n` : ''}${
      personalizedSnippet
        ? `Personalized Script Snapshot${
            shardAgeLabel ? ` (EN, curious, age ${shardAgeLabel})` : ' (EN, curious)'
          }: ${personalizedSnippet}\n`
        : ''
    }${vocabSummary ? `Key Terms: ${vocabSummary}\n` : ''}
`;
  });

  fs.writeFileSync(path.join(publicDir, 'llms-full.txt'), llmsFull);
  console.log('Generated llms-full.txt');

  // Generate minimal crawlable HTML lesson pages for search engines and AI
  lessons.forEach((lesson) => {
    const day = lesson.day_number;
    const topic = lesson.topic || 'Life Skills';
    const universalTruth = lesson.universal_truth || '';
    const marketingHeadline = lesson.marketing_headline || '';
    const marketingTagline = lesson.marketing_tagline || '';
    const marketingPitch = lesson.marketing_pitch || '';
    const mastery = lesson.mastery_criteria || '';

    const shard = shardByCoreLessonId.get(lesson.id);
    let personalizedSnippet = '';
    let vocabSummary = '';
    let shardAgeLabel = '';

    if (shard && shard.script_content) {
      const sc = shard.script_content;
      if (sc && typeof sc.script === 'string') {
        const scriptText = sc.script.trim();
        personalizedSnippet =
          scriptText.length > 320 ? `${scriptText.slice(0, 317)}...` : scriptText;
      }

      const vocabTerms =
        sc &&
        sc.vocabulary &&
        Array.isArray(sc.vocabulary.keyTerms) &&
        sc.vocabulary.keyTerms.length > 0
          ? sc.vocabulary.keyTerms.slice(0, 5)
          : [];

      if (vocabTerms.length > 0) {
        vocabSummary = vocabTerms.join(', ');
      }

      if (typeof shard.age === 'number') {
        shardAgeLabel = `${shard.age}`;
      }
    }

    const displayTitle = marketingHeadline || topic || `Day ${day} Lesson`;
    const metaDescription =
      marketingPitch ||
      marketingTagline ||
      universalTruth ||
      (personalizedSnippet
        ? personalizedSnippet
        : `Day ${day} – ${topic}. Personalized daily lesson with Kelly, your AI teacher, for learners ages 2-102.`);

    const html = `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Day ${day}: ${displayTitle} | Curious Kelly</title>
    <meta name="description" content="${metaDescription.replace(/"/g, '&quot;')}">
    <link rel="canonical" href="https://www.curiouskelly.com/learn.html?day=${day}">
  </head>
  <body>
    <h1>Day ${day}: ${displayTitle}</h1>
    <p><strong>Topic:</strong> ${topic}</p>
    ${
      universalTruth
        ? `<h2>Universal Truth</h2>
    <p>${universalTruth}</p>
    `
        : ''
    }${
      marketingTagline
        ? `<h2>Tagline</h2>
    <p>${marketingTagline}</p>
    `
        : ''
    }${
      marketingPitch
        ? `<h2>Lesson Pitch</h2>
    <p>${marketingPitch}</p>
    `
        : ''
    }${
      mastery
        ? `<h2>Mastery Criteria</h2>
    <p>${mastery}</p>
    `
        : ''
    }${
      personalizedSnippet
        ? `<h2>In Kelly's Words${
            shardAgeLabel ? ` (EN, curious, age ${shardAgeLabel})` : ' (EN, curious)'
          }</h2>
    <p>${personalizedSnippet}</p>
    `
        : ''
    }${
      vocabSummary
        ? `<h2>Key Vocabulary</h2>
    <p>${vocabSummary}</p>
    `
        : ''
    }    <p>Kelly, a photorealistic digital human teacher, guides learners ages 2-102 through this daily lesson as part of the 365-day Curious Kelly curriculum.</p>
    <p>To experience this lesson with Kelly in the full interactive player, visit
      <a href="https://www.curiouskelly.com/learn.html?day=${day}">https://www.curiouskelly.com/learn.html?day=${day}</a>.
    </p>
  </body>
</html>
`;

    const filePath = path.join(crawlableDir, `lesson-${day}.html`);
    fs.writeFileSync(filePath, html);
  });

  console.log('Generated crawlable HTML lesson pages in /public/crawlable');
}

generateSitemapsAndLlms();


