/**
 * Phase 6: Search & Discovery
 * 6A: Build full-text search index on core_lessons_v2
 * 6B: Generate thematic clusters (25 clusters, 365 lesson assignments)
 * 6C: Generate learning paths (20 curated sequences)
 */
require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

function log(msg) {
  console.log(`[${new Date().toISOString()}] DISCOVER | ${msg}`);
}

// ========== TASK 6A: Full-Text Search Index ==========

async function buildSearchIndex(client) {
  log('6A: Building Full-Text Search Index');

  // Check what columns exist
  const cols = await client.query(`
    SELECT column_name FROM information_schema.columns
    WHERE table_name = 'core_lessons_v2'
  `);
  const colNames = cols.rows.map(r => r.column_name);
  log(`6A: core_lessons_v2 columns: ${colNames.join(', ')}`);

  // Build search vector from available columns
  const vectorParts = [];
  if (colNames.includes('title')) vectorParts.push("setweight(to_tsvector('english', COALESCE(title, '')), 'A')");
  if (colNames.includes('subject')) vectorParts.push("setweight(to_tsvector('english', COALESCE(subject, '')), 'A')");
  if (colNames.includes('learning_objective')) vectorParts.push("setweight(to_tsvector('english', COALESCE(learning_objective, '')), 'B')");
  if (colNames.includes('category')) vectorParts.push("setweight(to_tsvector('english', COALESCE(category, '')), 'B')");

  if (vectorParts.length === 0) {
    log('6A: No text columns found, skipping');
    return 0;
  }

  const vectorExpr = vectorParts.join(' || ');

  try {
    await client.query(`UPDATE core_lessons_v2 SET search_vector = ${vectorExpr}`);
    log('6A: Search vectors populated');
  } catch (e) {
    log(`6A: Vector populate error: ${e.message}`);
  }

  try {
    await client.query(`CREATE INDEX IF NOT EXISTS idx_core_lessons_search ON core_lessons_v2 USING gin(search_vector)`);
    log('6A: GIN index created');
  } catch (e) {
    log(`6A: Index error: ${e.message}`);
  }

  // Test search
  try {
    const test = await client.query(
      "SELECT day_number, title FROM core_lessons_v2 WHERE search_vector @@ to_tsquery('english', 'science') LIMIT 5"
    );
    log(`6A: Test search 'science' found ${test.rows.length} results`);
  } catch (e) {
    log(`6A: Test search error: ${e.message}`);
  }

  const count = await client.query("SELECT COUNT(*) as cnt FROM core_lessons_v2 WHERE search_vector IS NOT NULL");
  log(`6A DONE: ${count.rows[0].cnt} lessons indexed`);
  return parseInt(count.rows[0].cnt);
}

// ========== TASK 6B: Thematic Clusters ==========

async function generateClusters(client) {
  log('6B: Generating Thematic Clusters');

  const existingClusters = await client.query('SELECT COUNT(*) as cnt FROM kellyos_clusters');
  if (parseInt(existingClusters.rows[0].cnt) >= 20) {
    log('6B: SKIP — clusters already generated');
    return parseInt(existingClusters.rows[0].cnt);
  }

  // Get all lessons
  const lessons = await client.query(
    'SELECT day_number, title, subject, category FROM core_lessons_v2 ORDER BY day_number'
  );
  const lessonList = lessons.rows.map(r => `Day ${r.day_number}: ${r.title} (${r.subject || r.category || 'general'})`).join('\n');

  try {
    const res = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [
          {
            role: 'system',
            content: `Analyze this curriculum of 365 daily lessons and group them into 25 thematic clusters. Each lesson should belong to 1-2 clusters. Return JSON: {
  "clusters": [
    {
      "name": "How the Universe Works",
      "description": "Lessons about space, astronomy, physics, and cosmology",
      "icon": "🌌",
      "color": "#1a1a2e",
      "lessons": [{"day": N, "relevance": 0.0-1.0}, ...]
    }
  ]
}`
          },
          { role: 'user', content: lessonList }
        ],
        temperature: 0.3,
        max_tokens: 16000,
        response_format: { type: 'json_object' }
      })
    });

    if (!res.ok) throw new Error(`OpenAI ${res.status}`);
    const data = await res.json();
    const result = JSON.parse(data.choices[0].message.content);

    let clusterCount = 0;
    let assignmentCount = 0;

    for (const cluster of (result.clusters || [])) {
      // Insert cluster
      const clusterRes = await client.query(`
        INSERT INTO kellyos_clusters (cluster_name, cluster_description, icon, color)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (cluster_name) DO UPDATE SET cluster_description = EXCLUDED.cluster_description
        RETURNING id
      `, [cluster.name, cluster.description, cluster.icon, cluster.color]);

      const clusterId = clusterRes.rows[0].id;
      clusterCount++;

      // Insert lesson assignments
      for (const lesson of (cluster.lessons || [])) {
        if (lesson.day >= 1 && lesson.day <= 365) {
          try {
            await client.query(`
              INSERT INTO kellyos_cluster_lessons (cluster_id, day_number, relevance_score)
              VALUES ($1, $2, $3)
              ON CONFLICT (cluster_id, day_number) DO UPDATE SET relevance_score = EXCLUDED.relevance_score
            `, [clusterId, lesson.day, lesson.relevance || 1.0]);
            assignmentCount++;
          } catch {}
        }
      }
    }

    log(`6B DONE: ${clusterCount} clusters, ${assignmentCount} assignments`);
    return clusterCount;
  } catch (e) {
    log(`6B ERROR: ${e.message}`);
    return 0;
  }
}

// ========== TASK 6C: Learning Paths ==========

async function generateLearningPaths(client) {
  log('6C: Generating Learning Paths');

  const existingPaths = await client.query('SELECT COUNT(*) as cnt FROM kellyos_learning_paths');
  if (parseInt(existingPaths.rows[0].cnt) >= 15) {
    log('6C: SKIP — paths already generated');
    return parseInt(existingPaths.rows[0].cnt);
  }

  const lessons = await client.query(
    'SELECT day_number, title, subject, category FROM core_lessons_v2 ORDER BY day_number'
  );
  const lessonList = lessons.rows.map(r => `Day ${r.day_number}: ${r.title} (${r.subject || r.category || 'general'})`).join('\n');

  try {
    const res = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [
          {
            role: 'system',
            content: `Create 20 curated learning paths from this curriculum. Each path is a sequence of 5-10 lessons that build on each other. Include paths for different interests and levels. Return JSON: {
  "paths": [
    {
      "name": "Space Explorer",
      "description": "Journey from our solar system to the edge of the universe",
      "difficulty": "beginner|intermediate|advanced",
      "age_range": "all ages|kids (5+)|teens (13+)|adults",
      "icon": "🚀",
      "lessons": [1, 15, 42, 88, 120, 155, 200, 250, 300, 345]
    }
  ]
}`
          },
          { role: 'user', content: lessonList }
        ],
        temperature: 0.4,
        max_tokens: 8000,
        response_format: { type: 'json_object' }
      })
    });

    if (!res.ok) throw new Error(`OpenAI ${res.status}`);
    const data = await res.json();
    const result = JSON.parse(data.choices[0].message.content);

    let pathCount = 0;
    for (const p of (result.paths || [])) {
      await client.query(`
        INSERT INTO kellyos_learning_paths (path_name, path_description, difficulty, age_range, estimated_days, icon, lessons)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (path_name) DO UPDATE SET
          path_description = EXCLUDED.path_description,
          lessons = EXCLUDED.lessons
      `, [
        p.name,
        p.description,
        p.difficulty || 'beginner',
        p.age_range || 'all ages',
        (p.lessons || []).length,
        p.icon || '📚',
        JSON.stringify(p.lessons || [])
      ]);
      pathCount++;
    }

    log(`6C DONE: ${pathCount} learning paths created`);
    return pathCount;
  } catch (e) {
    log(`6C ERROR: ${e.message}`);
    return 0;
  }
}

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  log('Connected. Starting search & discovery...');

  const results = {};
  results.search = await buildSearchIndex(client);
  results.clusters = await generateClusters(client);
  results.paths = await generateLearningPaths(client);

  log('=== SEARCH & DISCOVERY RESULTS ===');
  log(`Search indexed: ${results.search}`);
  log(`Clusters: ${results.clusters}`);
  log(`Learning paths: ${results.paths}`);

  const auditDir = path.join(__dirname, '..', 'kelly-pipeline', 'audit');
  if (!fs.existsSync(auditDir)) fs.mkdirSync(auditDir, { recursive: true });
  fs.writeFileSync(
    path.join(auditDir, 'discovery-results.json'),
    JSON.stringify({ timestamp: new Date().toISOString(), results }, null, 2)
  );

  await client.end();
}

main().catch(e => { console.error('[DISCOVER ERROR]', e); process.exit(1); });
