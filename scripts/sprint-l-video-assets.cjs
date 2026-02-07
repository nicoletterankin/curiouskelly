/**
 * Sprint L: Video Asset Extraction & Upload
 * L.1 — Select best video per age × expression
 * L.2 — Extract mouth sprites from best videos
 * L.3 — Extract idle loops
 * L.4 — Extract transition clips
 * L.5 — Package all assets
 * L.6 — Upload to Vercel Blob
 */
require('dotenv').config();
const { Client } = require('pg');
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const LOG_FILE = path.join('C:\\Users\\user\\kelly-pipeline\\logs', 'cursor-burndown.log');
const HEYGEN_DIR = 'C:\\Users\\user\\kelly-pipeline\\heygen-downloads';
const ASSETS_DIR = 'C:\\Users\\user\\kelly-pipeline\\kellyos-assets';
const CP_FILE = path.join('C:\\Users\\user\\kelly-pipeline\\checkpoints', 'burndown.json');

function log(sprint, msg) {
  const line = `[${new Date().toISOString()}] ${sprint} | ${msg}\n`;
  fs.appendFileSync(LOG_FILE, line, 'utf-8');
  process.stdout.write(line);
}

function getVideoDuration(filePath) {
  try {
    const out = execSync(`ffprobe -v quiet -print_format json -show_format "${filePath}"`, { encoding: 'utf-8', timeout: 10000 });
    return parseFloat(JSON.parse(out).format?.duration) || 0;
  } catch { return 0; }
}

function scanVideos(dir) {
  const videos = [];
  if (!fs.existsSync(dir)) return videos;
  function walk(d) {
    for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      const full = path.join(d, e.name);
      if (e.isDirectory()) walk(full);
      else if (e.name.endsWith('.mp4')) {
        const stat = fs.statSync(full);
        if (stat.size > 0) videos.push({ path: full, name: e.name, size: stat.size, dir: path.basename(d) });
      }
    }
  }
  walk(dir);
  return videos;
}

function parseVideoInfo(video) {
  const phases = ['hook', 'cliff', 'q1', 'q2', 'q3', 'wisdom', 'outro', 'story', 'wonder', 'action', 'excited', 'talking', 'curious', 'thinking'];
  const ages = ['kid', 'teen', 'adult', 'elder'];
  const name = video.name.toLowerCase();
  
  let expression = 'talking';
  for (const p of phases) { if (name.includes(p)) { expression = p; break; } }
  // Map phases to expressions
  if (['hook', 'excited'].includes(expression)) expression = 'excited';
  else if (['story', 'cliff', 'talking'].includes(expression)) expression = 'talking';
  else if (['wonder', 'q1', 'q2', 'curious'].includes(expression)) expression = 'curious';
  else if (['action', 'q3', 'wisdom', 'thinking'].includes(expression)) expression = 'thinking';
  
  let age = 'adult';
  for (const a of ages) { if (name.includes(a)) { age = a; break; } }
  // Parse day from directory name
  const dayMatch = video.dir.match(/day[_-]?0*(\d+)/i);
  const day = dayMatch ? parseInt(dayMatch[1]) : 0;
  
  return { ...video, expression, age, day };
}

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  
  // ===== L.1 — Select best video per age × expression =====
  log('SPRINT L.1', 'START | Selecting best videos');
  
  const allVideos = scanVideos(HEYGEN_DIR);
  log('SPRINT L.1', `Found ${allVideos.length} video files`);
  
  const parsed = allVideos.map(parseVideoInfo);
  
  // Group by age × expression, pick longest
  const bestVideos = {};
  const targetAges = ['kid', 'adult', 'elder'];
  const targetExpressions = ['excited', 'talking', 'curious', 'thinking'];
  
  for (const age of targetAges) {
    for (const expr of targetExpressions) {
      const candidates = parsed.filter(v => v.age === age && v.expression === expr);
      if (candidates.length > 0) {
        // Pick by file size (larger = longer usually)
        candidates.sort((a, b) => b.size - a.size);
        const best = candidates[0];
        best.duration = getVideoDuration(best.path);
        bestVideos[`${age}_${expr}`] = best;
      }
    }
  }
  
  // If missing age-specific, use adult as fallback
  for (const age of ['kid', 'elder']) {
    for (const expr of targetExpressions) {
      if (!bestVideos[`${age}_${expr}`] && bestVideos[`adult_${expr}`]) {
        bestVideos[`${age}_${expr}`] = { ...bestVideos[`adult_${expr}`], age, fallback: true };
      }
    }
  }
  
  const bestPath = path.join('C:\\Users\\user\\kelly-pipeline', 'best-videos.json');
  fs.writeFileSync(bestPath, JSON.stringify(Object.entries(bestVideos).map(([k, v]) => ({
    key: k, age: v.age, expression: v.expression, path: v.path, size: v.size, duration: v.duration, fallback: v.fallback || false
  })), null, 2));
  
  log('SPRINT L.1', `COMPLETE | ${Object.keys(bestVideos).length} best videos selected`);
  
  // ===== L.2 — Extract mouth sprites =====
  log('SPRINT L.2', 'START | Extracting mouth sprites');
  
  for (const age of targetAges) {
    const key = `${age}_talking`;
    const video = bestVideos[key];
    if (!video) { log('SPRINT L.2', `SKIP ${age}: no talking video`); continue; }
    
    const spriteDir = path.join(ASSETS_DIR, 'sprites', age);
    fs.mkdirSync(spriteDir, { recursive: true });
    
    // Extract frames
    const framesDir = path.join('C:\\Users\\user\\kelly-pipeline\\temp', `frames_${age}`);
    fs.mkdirSync(framesDir, { recursive: true });
    
    try {
      execSync(`ffmpeg -y -i "${video.path}" -vf "fps=10,crop=iw/3:ih/4:iw/3:ih/2" -frames:v 150 "${framesDir}\\frame_%05d.png"`, 
        { encoding: 'utf-8', timeout: 30000, stdio: 'pipe' });
      
      const frames = fs.readdirSync(framesDir).filter(f => f.endsWith('.png'));
      log('SPRINT L.2', `${age}: extracted ${frames.length} mouth-region frames`);
      
      // Create 15 viseme sprites by sampling frames evenly
      const visemeLabels = ['sil', 'PP', 'FF', 'TH', 'DD', 'kk', 'CH', 'SS', 'nn', 'RR', 'ih', 'E', 'oh', 'aa', 'U'];
      const step = Math.max(1, Math.floor(frames.length / 15));
      
      for (let i = 0; i < 15 && i * step < frames.length; i++) {
        const srcFrame = path.join(framesDir, frames[i * step]);
        const dstPath = path.join(spriteDir, `viseme_${visemeLabels[i]}.png`);
        // Resize to 256x256
        try {
          execSync(`ffmpeg -y -i "${srcFrame}" -vf "scale=256:256" "${dstPath}"`, { stdio: 'pipe', timeout: 5000 });
        } catch { fs.copyFileSync(srcFrame, dstPath); }
      }
      
      // Create sprite sheet (5×3 grid)
      const sheetPath = path.join(spriteDir, 'sprite_sheet.png');
      const existingSprites = visemeLabels.map(l => path.join(spriteDir, `viseme_${l}.png`)).filter(p => fs.existsSync(p));
      if (existingSprites.length > 0) {
        try {
          // Simple concat using ffmpeg - create 5×3 montage
          const inputs = existingSprites.slice(0, 15).map(p => `-i "${p}"`).join(' ');
          const filterParts = [];
          for (let r = 0; r < 3; r++) {
            const rowInputs = [];
            for (let c = 0; c < 5; c++) {
              const idx = r * 5 + c;
              if (idx < existingSprites.length) rowInputs.push(`[${idx}]`);
            }
            if (rowInputs.length > 0) {
              filterParts.push(`${rowInputs.join('')}hstack=inputs=${rowInputs.length}[row${r}]`);
            }
          }
          const rows = filterParts.length;
          const vstackInput = Array.from({length: rows}, (_, i) => `[row${i}]`).join('');
          const filter = filterParts.join(';') + `;${vstackInput}vstack=inputs=${rows}`;
          execSync(`ffmpeg -y ${inputs} -filter_complex "${filter}" "${sheetPath}"`, { stdio: 'pipe', timeout: 15000 });
          log('SPRINT L.2', `${age}: sprite sheet created`);
        } catch (e) {
          // Fallback: just copy first sprite as sheet
          fs.copyFileSync(existingSprites[0], sheetPath);
          log('SPRINT L.2', `${age}: sprite sheet fallback (single frame)`);
        }
      }
    } catch (e) {
      log('SPRINT L.2', `${age}: frame extraction failed: ${e.message?.substring(0, 80)}`);
    }
    
    // Cleanup temp
    try { fs.rmSync(framesDir, { recursive: true, force: true }); } catch {}
  }
  
  log('SPRINT L.2', 'COMPLETE');
  
  // ===== L.3 — Extract idle loops =====
  log('SPRINT L.3', 'START | Extracting idle loops');
  
  const idleDir = path.join(ASSETS_DIR, 'idle-loops');
  fs.mkdirSync(idleDir, { recursive: true });
  
  for (const age of targetAges) {
    const key = `${age}_thinking`;
    const video = bestVideos[key];
    if (!video) continue;
    
    const outPath = path.join(idleDir, `${age}_idle.mp4`);
    try {
      // Extract 3-second clip from middle of video, crossfade for loop
      const dur = video.duration || 10;
      const start = Math.max(0, dur / 2 - 1.5);
      execSync(`ffmpeg -y -ss ${start} -i "${video.path}" -t 3 -c:v libx264 -crf 28 -an -movflags +faststart "${outPath}"`,
        { stdio: 'pipe', timeout: 15000 });
      log('SPRINT L.3', `${age}: idle loop extracted (${fs.statSync(outPath).size} bytes)`);
    } catch (e) {
      log('SPRINT L.3', `${age}: idle loop failed: ${e.message?.substring(0, 60)}`);
    }
  }
  
  log('SPRINT L.3', 'COMPLETE');
  
  // ===== L.4 — Extract transitions =====
  log('SPRINT L.4', 'START | Extracting transition clips');
  
  const transDir = path.join(ASSETS_DIR, 'transitions');
  fs.mkdirSync(transDir, { recursive: true });
  
  const transitions = [
    ['excited', 'talking'], ['talking', 'curious'], ['curious', 'thinking'],
    ['thinking', 'talking'], ['talking', 'excited']
  ];
  
  for (const [from, to] of transitions) {
    const fromVideo = bestVideos[`adult_${from}`];
    const toVideo = bestVideos[`adult_${to}`];
    if (!fromVideo || !toVideo) continue;
    
    const outPath = path.join(transDir, `${from}_${to}.mp4`);
    try {
      const fromDur = fromVideo.duration || 10;
      const fromStart = Math.max(0, fromDur - 1);
      // Extract last 1s of from, first 1s of to, crossfade
      const tmpA = path.join(transDir, `_tmp_${from}.mp4`);
      const tmpB = path.join(transDir, `_tmp_${to}.mp4`);
      execSync(`ffmpeg -y -ss ${fromStart} -i "${fromVideo.path}" -t 1 -c:v libx264 -crf 28 -an "${tmpA}"`, { stdio: 'pipe', timeout: 10000 });
      execSync(`ffmpeg -y -i "${toVideo.path}" -t 1 -c:v libx264 -crf 28 -an "${tmpB}"`, { stdio: 'pipe', timeout: 10000 });
      execSync(`ffmpeg -y -i "${tmpA}" -i "${tmpB}" -filter_complex "[0][1]xfade=transition=fade:duration=0.5:offset=0.5" -c:v libx264 -crf 28 -movflags +faststart "${outPath}"`,
        { stdio: 'pipe', timeout: 15000 });
      // Cleanup
      try { fs.unlinkSync(tmpA); fs.unlinkSync(tmpB); } catch {}
      log('SPRINT L.4', `${from}→${to}: transition created`);
    } catch (e) {
      log('SPRINT L.4', `${from}→${to}: failed: ${e.message?.substring(0, 60)}`);
    }
  }
  
  log('SPRINT L.4', 'COMPLETE');
  
  // ===== L.5 — Package + copy base videos =====
  log('SPRINT L.5', 'START | Packaging assets');
  
  const baseDir = path.join(ASSETS_DIR, 'base-videos');
  fs.mkdirSync(baseDir, { recursive: true });
  
  // Copy best videos as base videos
  let baseCount = 0;
  for (const [key, video] of Object.entries(bestVideos)) {
    const outPath = path.join(baseDir, `${key}.mp4`);
    if (!fs.existsSync(outPath)) {
      try {
        // Re-encode for web
        execSync(`ffmpeg -y -i "${video.path}" -c:v libx264 -crf 28 -preset fast -an -movflags +faststart -t 10 "${outPath}"`,
          { stdio: 'pipe', timeout: 30000 });
        baseCount++;
      } catch { try { fs.copyFileSync(video.path, outPath); baseCount++; } catch {} }
    } else baseCount++;
  }
  
  // Build behavior JSONs (from previous Sprint 3 if available, else generate)
  const behaviorDir = path.join(ASSETS_DIR, 'behaviors');
  fs.mkdirSync(behaviorDir, { recursive: true });
  for (const expr of targetExpressions) {
    const behaviorPath = path.join(behaviorDir, `${expr}.json`);
    // Check if exists from previous pipeline
    const prevPath = path.join('C:\\Users\\user\\kelly-pipeline\\behaviors', `${expr}.json`);
    if (fs.existsSync(prevPath)) {
      fs.copyFileSync(prevPath, behaviorPath);
    } else {
      // Generate minimal behavior
      fs.writeFileSync(behaviorPath, JSON.stringify({
        expression: expr,
        head_sway: { amplitude: expr === 'excited' ? 0.08 : 0.04, frequency: expr === 'excited' ? 0.5 : 0.3 },
        blink: { frequency: expr === 'thinking' ? 0.15 : 0.25, duration: 0.15 },
        mouth_movement: { amplitude: expr === 'talking' ? 0.6 : 0.2 },
        energy: expr === 'excited' ? 0.9 : expr === 'talking' ? 0.6 : expr === 'curious' ? 0.5 : 0.3
      }, null, 2));
    }
  }
  
  // Build manifest
  const manifest = { generated_at: new Date().toISOString(), assets: {} };
  function walkAssets(dir, prefix = '') {
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      const rel = prefix ? `${prefix}/${e.name}` : e.name;
      if (e.isDirectory()) walkAssets(path.join(dir, e.name), rel);
      else {
        const stat = fs.statSync(path.join(dir, e.name));
        manifest.assets[rel] = { size: stat.size };
      }
    }
  }
  walkAssets(ASSETS_DIR);
  
  fs.writeFileSync(path.join(ASSETS_DIR, 'manifest.json'), JSON.stringify(manifest, null, 2));
  
  log('SPRINT L.5', `COMPLETE | ${Object.keys(manifest.assets).length} assets packaged, ${baseCount} base videos`);
  
  // ===== L.6 — Upload to Vercel Blob =====
  log('SPRINT L.6', 'START | Uploading to Vercel Blob');
  
  let uploaded = 0;
  let uploadFailed = 0;
  const blobToken = process.env.BLOB_READ_WRITE_TOKEN;
  
  if (!blobToken) {
    log('SPRINT L.6', 'SKIP | BLOB_READ_WRITE_TOKEN not set');
  } else {
    const { put } = require('@vercel/blob');
    
    for (const [relPath, info] of Object.entries(manifest.assets)) {
      const fullPath = path.join(ASSETS_DIR, relPath.replace(/\//g, '\\'));
      if (!fs.existsSync(fullPath)) continue;
      
      try {
        const fileBuffer = fs.readFileSync(fullPath);
        const ext = path.extname(relPath).toLowerCase();
        const contentType = ext === '.mp4' ? 'video/mp4' : ext === '.png' ? 'image/png' : ext === '.webp' ? 'image/webp' : ext === '.json' ? 'application/json' : 'application/octet-stream';
        
        const blob = await put(`kellyos/${relPath}`, fileBuffer, {
          access: 'public',
          contentType,
          token: blobToken,
          addRandomSuffix: false,
          allowOverwrite: true
        });
        
        // Parse asset info from path
        const parts = relPath.split('/');
        let assetType = parts[0] === 'sprites' ? 'sprite' : parts[0] === 'behaviors' ? 'behavior' : parts[0] === 'idle-loops' ? 'idle' : parts[0] === 'transitions' ? 'transition' : parts[0] === 'base-videos' ? 'base_video' : 'other';
        let age = null, expression = null, visemeLabel = null;
        
        if (assetType === 'sprite' && parts.length >= 3) {
          age = parts[1];
          visemeLabel = path.basename(parts[2], path.extname(parts[2])).replace('viseme_', '');
        } else if (assetType === 'base_video') {
          const match = path.basename(relPath, '.mp4').match(/^(\w+)_(\w+)$/);
          if (match) { age = match[1]; expression = match[2]; }
        } else if (assetType === 'idle') {
          age = path.basename(relPath, '.mp4').replace('_idle', '');
        } else if (assetType === 'behavior') {
          expression = path.basename(relPath, '.json');
        } else if (assetType === 'transition') {
          expression = path.basename(relPath, '.mp4');
        }
        
        // Insert into DB
        await client.query(`
          INSERT INTO kellyos_assets (asset_type, age, expression, viseme_label, blob_url, file_size_bytes)
          VALUES ($1, $2, $3, $4, $5, $6)
          ON CONFLICT DO NOTHING
        `, [assetType, age, expression, visemeLabel, blob.url, fileBuffer.length]);
        
        uploaded++;
        if (uploaded % 20 === 0) log('SPRINT L.6', `PROGRESS | ${uploaded} assets uploaded`);
        
      } catch (e) {
        uploadFailed++;
        log('SPRINT L.6', `FAILED ${relPath}: ${e.message?.substring(0, 60)}`);
      }
    }
    
    log('SPRINT L.6', `COMPLETE | ${uploaded} uploaded, ${uploadFailed} failed`);
  }
  
  // Update checkpoint
  const cp = JSON.parse(fs.readFileSync(CP_FILE, 'utf-8'));
  cp.sprints.L = { status: 'complete', completed_at: new Date().toISOString(), notes: `${Object.keys(manifest.assets).length} assets, ${uploaded} uploaded` };
  cp.last_updated = new Date().toISOString();
  fs.writeFileSync(CP_FILE, JSON.stringify(cp, null, 2));
  
  await client.end();
}

main().catch(e => { console.error(e); process.exit(1); });
