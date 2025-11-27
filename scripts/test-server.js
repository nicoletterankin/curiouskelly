/**
 * Curious Kelly - Local Test Server
 * 
 * Provides API endpoints for the test dashboard to verify:
 * - Environment configuration
 * - Supabase connectivity
 * - ElevenLabs voice generation
 * - Expression generation
 * - Kelly Engine integration
 * 
 * Run with: node scripts/test-server.js
 */

import { createRequire } from 'module';
import { resolve, dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { existsSync, readFileSync } from 'fs';
import http from 'http';

// Setup paths
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = resolve(__dirname, '..');

// Try to load dependencies
let createClient, dotenv;

// Try multiple paths for dependencies
const possibleNodeModules = [
  resolve(projectRoot, 'node_modules'),
  resolve(projectRoot, 'daily-lesson-marketing', 'node_modules'),
  resolve(projectRoot, 'curiouskelly-marketing-site', 'node_modules'),
];

let modulePath = null;
for (const path of possibleNodeModules) {
  if (existsSync(resolve(path, '@supabase', 'supabase-js'))) {
    modulePath = path;
    break;
  }
}

if (modulePath) {
  const require = createRequire(resolve(modulePath, '..', 'package.json'));
  try {
    const supabaseModule = require('@supabase/supabase-js');
    createClient = supabaseModule.createClient;
    dotenv = require('dotenv');
  } catch (e) {
    console.warn('Could not load from existing node_modules:', e.message);
  }
}

// Load environment variables
const envPath = resolve(projectRoot, '.env');
if (dotenv && existsSync(envPath)) {
  dotenv.config({ path: envPath });
  console.log('✅ Loaded .env from:', envPath);
} else {
  console.warn('⚠️  .env file not found or dotenv not available');
}

// Server configuration
const PORT = process.env.TEST_SERVER_PORT || 3000;

// MIME types for static files
const MIME_TYPES = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.mjs': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.wasm': 'application/wasm',
  '.br': 'application/octet-stream',
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.webm': 'video/webm',
};

// ============================================================================
// API HANDLERS
// ============================================================================

/**
 * Test environment configuration
 */
async function handleTestEnv(req, res) {
  const required = [
    'PUBLIC_SUPABASE_URL',
    'PUBLIC_SUPABASE_ANON_KEY',
    'ELEVENLABS_API_KEY',
    'ELEVENLABS_KELLY_VOICE_ID',
    'STRIPE_SECRET_KEY',
  ];
  
  const configured = {};
  const missing = [];
  
  for (const key of required) {
    if (process.env[key]) {
      configured[key] = '✅ Set';
    } else {
      configured[key] = '❌ Missing';
      missing.push(key);
    }
  }
  
  sendJson(res, {
    success: missing.length === 0,
    configured,
    missing,
    envPath: envPath,
    envExists: existsSync(envPath),
  });
}

/**
 * Test Supabase connection
 */
async function handleTestSupabase(req, res, urlParams) {
  if (!createClient) {
    sendJson(res, { 
      success: false, 
      error: 'Supabase client not available. Run: npm install @supabase/supabase-js' 
    });
    return;
  }

  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL;
  const supabaseKey = process.env.PUBLIC_SUPABASE_ANON_KEY;
  
  if (!supabaseUrl || !supabaseKey) {
    sendJson(res, { 
      success: false, 
      error: 'Missing PUBLIC_SUPABASE_URL or PUBLIC_SUPABASE_ANON_KEY in .env' 
    });
    return;
  }
  
  try {
    const supabase = createClient(supabaseUrl, supabaseKey);
    
    // Get lesson count
    const { count, error } = await supabase
      .from('core_lessons')
      .select('*', { count: 'exact', head: true });
    
    if (error) throw error;
    
    let result = { success: true, lessonCount: count };
    
    // Optionally fetch sample lesson
    if (urlParams.get('sample') === 'true') {
      const { data: sample } = await supabase
        .from('core_lessons')
        .select('*')
        .eq('day_number', 1)
        .single();
      
      result.sample = sample;
    }
    
    sendJson(res, result);
  } catch (error) {
    sendJson(res, { success: false, error: error.message });
  }
}

/**
 * Test ElevenLabs voice generation
 */
async function handleTestElevenLabs(req, res) {
  const apiKey = process.env.ELEVENLABS_API_KEY;
  const voiceId = process.env.ELEVENLABS_KELLY_VOICE_ID;
  
  if (!apiKey || !voiceId) {
    sendJson(res, { 
      success: false, 
      error: 'Missing ELEVENLABS_API_KEY or ELEVENLABS_KELLY_VOICE_ID in .env' 
    }, 400);
    return;
  }
  
  // Get request body
  const body = await getRequestBody(req);
  const text = body.text || "Hello! I'm Kelly, your curious learning companion!";
  
  try {
    const response = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
      {
        method: 'POST',
        headers: {
          'Accept': 'audio/mpeg',
          'Content-Type': 'application/json',
          'xi-api-key': apiKey
        },
        body: JSON.stringify({
          text,
          model_id: 'eleven_multilingual_v2',
          voice_settings: {
            stability: 0.5,
            similarity_boost: 0.75,
            style: 0.0,
            use_speaker_boost: true
          }
        })
      }
    );
    
    if (!response.ok) {
      const errorText = await response.text();
      let errorMessage = `HTTP ${response.status}`;
      try {
        const errorJson = JSON.parse(errorText);
        errorMessage = errorJson.detail?.message || errorJson.detail || errorMessage;
      } catch {
        errorMessage = errorText || errorMessage;
      }
      throw new Error(errorMessage);
    }
    
    const buffer = await response.arrayBuffer();
    
    res.writeHead(200, {
      'Content-Type': 'audio/mpeg',
      'Content-Length': buffer.byteLength,
    });
    res.end(Buffer.from(buffer));
    
  } catch (error) {
    sendJson(res, { success: false, error: error.message }, 500);
  }
}

/**
 * Test expression generation
 */
async function handleTestExpressions(req, res) {
  const body = await getRequestBody(req);
  
  const { text, archetype, tone, ageBucket, language, phase } = body;
  
  try {
    // Simulate expression generation (the real generator is browser-only ES modules)
    const expressions = generateMockExpressions(text, archetype, tone, ageBucket, phase);
    const gestures = generateMockGestures(archetype, phase);
    
    sendJson(res, {
      success: true,
      expressions,
      gestures,
      metadata: {
        archetype,
        tone,
        ageBucket,
        language,
        phase,
        textLength: text?.length || 0,
      }
    });
  } catch (error) {
    sendJson(res, { success: false, error: error.message });
  }
}

/**
 * Test Kelly Engine modules
 */
async function handleTestEngine(req, res) {
  const modules = [
    'elevenlabs-voice-engine.js',
    'expression-generator.js',
    'phase-loader.js',
    'cache-manager.js',
    'unity-loader.js',
    'unity-asset-manager.js',
    'unity-audio-coordinator.js',
    'unity-bridge.js',
    'supabase-service.js',
    'kelly-engine.js',
  ];
  
  const results = [];
  const missing = [];
  
  for (const module of modules) {
    const modulePath = resolve(projectRoot, 'app', module);
    if (existsSync(modulePath)) {
      results.push(module);
    } else {
      missing.push(module);
    }
  }
  
  sendJson(res, {
    success: missing.length === 0,
    modules: results,
    missing,
  });
}

/**
 * Test integration - fetch lesson
 */
async function handleTestIntegration(req, res, urlParams) {
  if (!createClient) {
    sendJson(res, { 
      success: false, 
      error: 'Supabase client not available' 
    });
    return;
  }

  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL;
  const supabaseKey = process.env.PUBLIC_SUPABASE_ANON_KEY;
  
  if (!supabaseUrl || !supabaseKey) {
    sendJson(res, { 
      success: false, 
      error: 'Missing Supabase credentials' 
    });
    return;
  }
  
  const day = parseInt(urlParams.get('day')) || 1;
  
  try {
    const supabase = createClient(supabaseUrl, supabaseKey);
    
    const { data: lesson, error } = await supabase
      .from('core_lessons')
      .select('*')
      .eq('day_number', day)
      .single();
    
    if (error) throw error;
    
    sendJson(res, {
      success: true,
      lesson: {
        id: lesson.id,
        day_number: lesson.day_number,
        topic: lesson.topic,
        universal_truth: lesson.universal_truth,
      }
    });
  } catch (error) {
    sendJson(res, { success: false, error: error.message });
  }
}

// ============================================================================
// MOCK DATA GENERATORS
// ============================================================================

function generateMockExpressions(text, archetype, tone, ageBucket, phase) {
  const baseIntensity = tone === 'enthusiastic' ? 0.8 : tone === 'serious' ? 0.5 : 0.6;
  const sentenceCount = (text?.match(/[.!?]/g) || []).length || 1;
  
  const expressions = [];
  for (let i = 0; i < Math.min(sentenceCount + 2, 8); i++) {
    expressions.push({
      timestamp: i * 3,
      emotion: ['curious', 'excited', 'explaining', 'warm', 'thoughtful'][i % 5],
      intensity: baseIntensity + (Math.random() * 0.2 - 0.1),
      blendShapes: {
        smile: 30 + Math.random() * 40,
        eyebrowRaise: 20 + Math.random() * 30,
        eyesWide: 15 + Math.random() * 25,
      },
      transitionDuration: 0.3,
    });
  }
  
  return expressions;
}

function generateMockGestures(archetype, phase) {
  const gestureLibrary = {
    'The Explorer': ['point_up_dramatic', 'arms_wide_open', 'reaching_forward'],
    'The Scientist': ['chin_touch', 'glasses_adjust', 'finger_point_precise'],
    'The Artist': ['hands_flowing', 'frame_gesture', 'heart_touch'],
    'The Storyteller': ['theatrical_pause', 'character_mime', 'expansive_reveal'],
    'default': ['nod', 'palm_up_single', 'gentle_gesture'],
  };
  
  const gestures = gestureLibrary[archetype] || gestureLibrary.default;
  
  return gestures.map((gesture, i) => ({
    timestamp: i * 5 + 1,
    gesture,
    duration: 1.5 + Math.random(),
    intensity: 0.6 + Math.random() * 0.3,
    context: phase,
  }));
}

// ============================================================================
// STATIC FILE SERVER
// ============================================================================

function serveStaticFile(req, res, urlPath) {
  // Map URL paths to file system
  let filePath;
  
  if (urlPath === '/' || urlPath === '/index.html') {
    filePath = resolve(projectRoot, 'public', 'app.html');
  } else if (urlPath === '/test' || urlPath === '/test-dashboard.html') {
    filePath = resolve(projectRoot, 'public', 'test-dashboard.html');
  } else if (urlPath.startsWith('/unity/')) {
    filePath = resolve(projectRoot, 'public', urlPath.substring(1));
  } else if (urlPath.startsWith('/app/')) {
    filePath = resolve(projectRoot, urlPath.substring(1));
  } else {
    filePath = resolve(projectRoot, 'public', urlPath.substring(1));
  }
  
  // Check if file exists
  if (!existsSync(filePath)) {
    // Try without leading slash
    const altPath = resolve(projectRoot, urlPath.substring(1));
    if (existsSync(altPath)) {
      filePath = altPath;
    } else {
      res.writeHead(404);
      res.end('Not Found');
      return;
    }
  }
  
  // Get MIME type
  const ext = '.' + filePath.split('.').pop().toLowerCase();
  const mimeType = MIME_TYPES[ext] || 'application/octet-stream';
  
  // Handle Brotli-compressed Unity files
  const headers = { 'Content-Type': mimeType };
  if (ext === '.br') {
    headers['Content-Encoding'] = 'br';
    // Determine actual content type from filename
    if (filePath.includes('.wasm')) {
      headers['Content-Type'] = 'application/wasm';
    } else if (filePath.includes('.js')) {
      headers['Content-Type'] = 'application/javascript';
    }
  }
  
  try {
    const content = readFileSync(filePath);
    res.writeHead(200, headers);
    res.end(content);
  } catch (error) {
    res.writeHead(500);
    res.end('Internal Server Error');
  }
}

// ============================================================================
// UTILITIES
// ============================================================================

function sendJson(res, data, statusCode = 200) {
  res.writeHead(statusCode, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(data));
}

async function getRequestBody(req) {
  return new Promise((resolve) => {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try {
        resolve(JSON.parse(body || '{}'));
      } catch {
        resolve({});
      }
    });
  });
}

// ============================================================================
// REQUEST ROUTER
// ============================================================================

async function handleRequest(req, res) {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const urlPath = url.pathname;
  const urlParams = url.searchParams;
  
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }
  
  // API routes
  if (urlPath.startsWith('/api/')) {
    const endpoint = urlPath.replace('/api/', '');
    
    switch (endpoint) {
      case 'test-env':
        return handleTestEnv(req, res);
      case 'test-supabase':
        return handleTestSupabase(req, res, urlParams);
      case 'test-elevenlabs':
        return handleTestElevenLabs(req, res);
      case 'test-expressions':
        return handleTestExpressions(req, res);
      case 'test-engine':
        return handleTestEngine(req, res);
      case 'test-integration':
        return handleTestIntegration(req, res, urlParams);
      default:
        sendJson(res, { error: 'Unknown API endpoint' }, 404);
        return;
    }
  }
  
  // Static files
  serveStaticFile(req, res, urlPath);
}

// ============================================================================
// START SERVER
// ============================================================================

const server = http.createServer(handleRequest);

server.listen(PORT, () => {
  console.log(`
  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║   🧪 CURIOUS KELLY TEST SERVER                               ║
  ║                                                              ║
  ╠══════════════════════════════════════════════════════════════╣
  ║                                                              ║
  ║   🌐 Dashboard:  http://localhost:${PORT}/test-dashboard.html    ║
  ║   📁 App:        http://localhost:${PORT}/                       ║
  ║   🎮 Unity:      http://localhost:${PORT}/unity/kelly-live/      ║
  ║                                                              ║
  ╠══════════════════════════════════════════════════════════════╣
  ║                                                              ║
  ║   API Endpoints:                                             ║
  ║   • GET  /api/test-env          - Check environment vars     ║
  ║   • GET  /api/test-supabase     - Test database connection   ║
  ║   • POST /api/test-elevenlabs   - Generate voice audio       ║
  ║   • POST /api/test-expressions  - Generate expressions       ║
  ║   • GET  /api/test-engine       - Verify Kelly Engine        ║
  ║   • GET  /api/test-integration  - Full integration test      ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
  `);
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`\n❌ Port ${PORT} is already in use.`);
    console.log(`   Try: set TEST_SERVER_PORT=3001 && node scripts/test-server.js`);
  } else {
    console.error('Server error:', err);
  }
  process.exit(1);
});



