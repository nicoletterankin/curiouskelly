/**
 * Sprint D: Audit HeyGen Video Inventory
 * Scan all video files, check integrity, categorize
 */
require('dotenv').config();
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const VIDEO_DIRS = [
  'C:\\Users\\user\\kelly-pipeline\\heygen-downloads',
  'C:\\Users\\user\\Downloads',
];

function probeVideo(filePath) {
  try {
    const cmd = `ffprobe -v quiet -print_format json -show_format -show_streams "${filePath}"`;
    const output = execSync(cmd, { encoding: 'utf-8', timeout: 10000 });
    const probe = JSON.parse(output);
    const videoStream = probe.streams?.find(s => s.codec_type === 'video');
    const audioStream = probe.streams?.find(s => s.codec_type === 'audio');
    
    return {
      valid: true,
      duration: parseFloat(probe.format?.duration) || 0,
      size: parseInt(probe.format?.size) || 0,
      codec: videoStream?.codec_name || 'unknown',
      resolution: videoStream ? `${videoStream.width}x${videoStream.height}` : 'unknown',
      fps: videoStream ? eval(videoStream.r_frame_rate) : 0,
      has_audio: !!audioStream,
      audio_codec: audioStream?.codec_name || null,
    };
  } catch (e) {
    return { valid: false, error: e.message.substring(0, 100) };
  }
}

function scanDirectory(dir) {
  const results = [];
  if (!fs.existsSync(dir)) return results;
  
  function walk(d) {
    const entries = fs.readdirSync(d, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(d, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
      } else if (entry.name.endsWith('.mp4') || entry.name.endsWith('.MP4')) {
        const stat = fs.statSync(fullPath);
        results.push({
          path: fullPath,
          filename: entry.name,
          relativePath: path.relative(dir, fullPath),
          size: stat.size,
        });
      }
    }
  }
  
  walk(dir);
  return results;
}

function parseFilename(filename) {
  // Pattern: phase-age.mp4 (e.g., hook-adult.mp4)
  const match1 = filename.match(/^(.+)-(.+)\.mp4$/i);
  if (match1) return { phase: match1[1], age: match1[2] };
  
  // Pattern: day_NNN_phase.mp4
  const match2 = filename.match(/day[_-]?(\d+)[_-](.+)\.mp4$/i);
  if (match2) return { day: parseInt(match2[1]), phase: match2[2] };
  
  // HeyGen pattern with UUID
  const match3 = filename.match(/([a-f0-9]{32})\.mp4$/i);
  if (match3) return { heygen_id: match3[1] };
  
  return { unparsed: filename };
}

async function main() {
  console.log('=== HeyGen Video Audit ===\n');
  
  const allVideos = [];
  let valid = 0, zeroBytes = 0, corrupted = 0, missingAudio = 0;
  
  for (const dir of VIDEO_DIRS) {
    console.log(`Scanning: ${dir}`);
    const files = scanDirectory(dir);
    console.log(`  Found ${files.length} MP4 files`);
    
    let processed = 0;
    for (const file of files) {
      if (file.size === 0) {
        zeroBytes++;
        allVideos.push({
          ...file,
          status: 'zero-byte',
          parsed: parseFilename(file.filename),
        });
      } else {
        const probe = probeVideo(file.path);
        if (probe.valid) {
          if (!probe.has_audio) {
            missingAudio++;
            allVideos.push({ ...file, ...probe, status: 'missing-audio', parsed: parseFilename(file.filename) });
          } else {
            valid++;
            allVideos.push({ ...file, ...probe, status: 'valid', parsed: parseFilename(file.filename) });
          }
        } else {
          corrupted++;
          allVideos.push({ ...file, status: 'corrupted', error: probe.error, parsed: parseFilename(file.filename) });
        }
      }
      
      processed++;
      if (processed % 50 === 0) {
        process.stdout.write(`  Probed ${processed}/${files.length}\r`);
      }
    }
    console.log(`  Probed ${files.length}/${files.length} complete`);
  }
  
  // Calculate totals
  const totalSize = allVideos.reduce((sum, v) => sum + (v.size || 0), 0);
  const totalDuration = allVideos.filter(v => v.valid).reduce((sum, v) => sum + (v.duration || 0), 0);
  
  const audit = {
    summary: {
      total: allVideos.length,
      valid,
      zero_byte: zeroBytes,
      corrupted,
      missing_audio: missingAudio,
      total_size_gb: Math.round(totalSize / 1024 / 1024 / 1024 * 100) / 100,
      total_duration_minutes: Math.round(totalDuration / 60 * 10) / 10,
    },
    videos: allVideos.map(v => ({
      filename: v.filename,
      relativePath: v.relativePath,
      status: v.status,
      size: v.size,
      duration: v.duration || 0,
      resolution: v.resolution || null,
      parsed: v.parsed,
    }))
  };
  
  // Write audit file
  const auditPath = path.join('C:\\Users\\user\\kelly-pipeline\\audit', 'heygen-video-audit.json');
  fs.writeFileSync(auditPath, JSON.stringify(audit, null, 2));
  
  console.log(`\n=== Audit Summary ===`);
  console.log(`Total MP4 files: ${allVideos.length}`);
  console.log(`Valid: ${valid}`);
  console.log(`Zero-byte: ${zeroBytes}`);
  console.log(`Corrupted: ${corrupted}`);
  console.log(`Missing audio: ${missingAudio}`);
  console.log(`Total size: ${audit.summary.total_size_gb} GB`);
  console.log(`Total duration: ${audit.summary.total_duration_minutes} minutes`);
  console.log(`\nAudit saved to: ${auditPath}`);
}

main().catch(e => { console.error(e); process.exit(1); });
