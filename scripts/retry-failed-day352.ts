#!/usr/bin/env npx tsx
/**
 * 🔄 RETRY FAILED DAY 352 VIDEOS
 * Retries only the failed archetypes from the overnight run
 */
import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';

const FAILED_ARCHETYPES = ['rebel', 'architect', 'macgyver', 'storyteller', 'strategist'];

const DAY_351_HEYGEN_URLS: Record<string, string> = {
  rebel: "https://files2.heygen.ai/aws_pacific/avatar_tmp/363d4ebe9e2f4abcb1ec4f962e339916/57bc3eb5ba9942589139de07a58ccb9e.mp4?Expires=1766597533&Signature=ej0pAouS~UUHF4vgkxAZmOP6jR7DXGQDxjYj9YcvgujE7VDELNMvFnaKhK2pM40IfimTZrWSpBQa9DzvOBCFbeGYlySEJVq2~nZqrYNZIo3A1XBycEC4KUxsnM7QKdgqcYKYvAr8GYAH~iYK1U9BoYMHTSw9e2qjJz1IfUWHU6~gMkacl-VKrW9~c7hVeFbORuuy8JTTONHeHb0cbCanDHEmYrWWiBORC-wQ~T7dhiK24Q6DnyeKWcDx~J75yVF6DbuKw-IvVyaKZxVvjUHBcUgx7ut1o4hywyVc6PLPo-RcB7o6WHCPXo4HVzt7Tld1OkeOwmHVIp6h1BHNXGy7uA__&Key-Pair-Id=K38HBHX5LX3X2H",
  architect: "https://files2.heygen.ai/aws_pacific/avatar_tmp/363d4ebe9e2f4abcb1ec4f962e339916/33b85429402f46c69d1dc147dbf539f1.mp4?Expires=1766602141&Signature=IG4TQ7hRnHsa5JchDxOMF4Xcl2mW8N0VupD0Y5Ni0iERWRIJhRScY3HmYQLfpQh6IMUE7GtKxlLejrDfuB2-q2x5t1e1hA34ZTZewNgJIuvqWQV6pGZ4xBavn8oN9k-jckrap8kdhEgbMUdSGTPXhDXkBxOFS14tTuNZ3cFzBsmqRHvdb7S9H5f6~9jq33MXqal3ziDMMR8EK9f8vCFhT2cK~i-PMAwehLK90ZRBNTvQNemXXgfdfCK7s0IZG-IaQmrrBheiPGEWiglecsSR~8Yi2aIPdBIaQcyLLxPwo7nrzON24UPrYokwOXliVLuzI19vL6DX1uMwZcQ0H3Dx1A__&Key-Pair-Id=K38HBHX5LX3X2H",
  macgyver: "https://files2.heygen.ai/aws_pacific/avatar_tmp/363d4ebe9e2f4abcb1ec4f962e339916/c80f30575d3d471dafedc42491fb93a9.mp4?Expires=1766600824&Signature=Dbc-OA7w5BhEQPfCzSyKhQ7~-MlCPs4zSBqvHkD8uVW6V5LMnNFQMxyEQP-vkvi8UcVFjzSsxE7AvyKmrLFdTo6rbyTabT-KEnSdLtIde6N8shL9m8nQZ88L~WNnT39uvb4HkmYoQpy4ZU~jVxA6DNg-tgL9V7Hqes8n~GjtEHl2pmjArhl6uLx8VvNyFowTWTWeE3np3iebhFsDlZyOxBX0wDYUr75zrqUE0H23Ho9bl~~pIf7g0Fv7uzW-K5ow9wCP7MJU6Zo3kU~mhohgiPvX6o3cYh9Q77UUC4vfKGwvQ3IRy1zQw07F1MMOa8IoWO-B-g~1~zTUgW80AFTmqg__&Key-Pair-Id=K38HBHX5LX3X2H",
  storyteller: "https://files2.heygen.ai/aws_pacific/avatar_tmp/363d4ebe9e2f4abcb1ec4f962e339916/dbffa94029f3453ea2570528cac49f61.mp4?Expires=1766614667&Signature=JsbOKRqSR0y7FlDSEpMuvQmVfG0cWwXa~K1VIEkyQAPxtywhAietZSz-ItcAI8aiyYH911UO5XQAz2Db8oXsFaFmypjHLD6e~PrV6RTtz7WED1969zZdk2A8bbVfQWC3w16kl17a9jaaBQrCdy6qSiUOKlKktfpdlwft2ITrRMFh4E8NFhc6d6CvNt6lZO2uoRefoq3hZHUluu7-7egqnRYrNGVuSutz1k7ajFYpBb15XEJ~AAblpL0xnF7fAhV4A0uS07Mio1h05M5llJ6qrlweELygnyd3mXiY-UJxsKvr-FjKVKqNvWc~RdzGwDdZ29x~wCGvU272L3Inmw2joQ__&Key-Pair-Id=K38HBHX5LX3X2H",
  strategist: "https://files2.heygen.ai/aws_pacific/avatar_tmp/363d4ebe9e2f4abcb1ec4f962e339916/574fb2f7881c47cdbd655cdc2833ab81.mp4?Expires=1766599239&Signature=boSdF8tup-hXa78hBfZiESc6LCi0Vh1mN5sjLybkZth-xSbixagPcSmQSAwE3Pu5RvM4mSAKcc1UgpL4MkpW8U4Um19XLIr3W8Lq9Vc1pcdq-lR5lkAoe5pEO8JNhEj4k-PFBNSkoqtp7gfFUZXQV5xvt7-dJWT0WMnjK43oHJ8h-oUmLXYyIR5eKsQ0VHYzJUSrFfEBgAiI0nCd1md5rsd~cTPkJ257lr6BTE0KAk7zhfKmQ9QN2c8hCoxqDwMbRyAcD~I--XpEJ-v~0pLyoG9THLwJBjuP8w38QWN1oPzvGkvBI2GSefPxZRK6mwYaFn7-ZdhC29qWqgN5NHEyGA__&Key-Pair-Id=K38HBHX5LX3X2H"
};

const KELLY_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';
const ELEVENLABS_API = 'https://api.elevenlabs.io/v1';
const SYNC_LABS_API = 'https://api.sync.so/v2';

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function getLesson(): Promise<any> {
  const content = fs.readFileSync('public/lessons/day-352.json', 'utf-8');
  return JSON.parse(content);
}

async function generateAudio(text: string, filename: string): Promise<Buffer> {
  console.log(`  🎤 Generating audio for ${filename}...`);
  const response = await fetch(`${ELEVENLABS_API}/text-to-speech/${KELLY_VOICE_ID}`, {
    method: 'POST',
    headers: {
      'xi-api-key': process.env.ELEVENLABS_API_KEY!,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text,
      model_id: 'eleven_turbo_v2_5',
      voice_settings: { stability: 0.5, similarity_boost: 0.75 }
    })
  });
  if (!response.ok) throw new Error(`ElevenLabs error: ${response.status}`);
  return Buffer.from(await response.arrayBuffer());
}

async function uploadAudio(buffer: Buffer, filename: string): Promise<string> {
  const path = `sync-labs-redub/${filename}`;
  const { error } = await supabase.storage
    .from('kelly-templates')
    .upload(path, buffer, { contentType: 'audio/mpeg', upsert: true });
  if (error) throw error;
  const { data } = supabase.storage.from('kelly-templates').getPublicUrl(path);
  return data.publicUrl;
}

async function redubWithSyncLabs(videoUrl: string, audioUrl: string): Promise<string> {
  console.log(`  🎬 Submitting to Sync Labs...`);
  const response = await fetch(`${SYNC_LABS_API}/generate`, {
    method: 'POST',
    headers: {
      'x-api-key': process.env.SYNC_LABS_API_KEY!,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'lipsync-2',
      input: [{ type: 'video', url: videoUrl }, { type: 'audio', url: audioUrl }],
      options: { output_format: 'mp4' }
    })
  });
  if (!response.ok) throw new Error(`Sync Labs error: ${response.status}`);
  const data = await response.json();
  const jobId = data.id;
  console.log(`  ⏳ Job ${jobId} - polling...`);

  // Poll with longer timeout (10 min)
  const maxWait = 600000;
  const start = Date.now();
  while (Date.now() - start < maxWait) {
    await new Promise(r => setTimeout(r, 10000)); // 10s intervals
    const status = await fetch(`${SYNC_LABS_API}/generate/${jobId}`, {
      headers: { 'x-api-key': process.env.SYNC_LABS_API_KEY! }
    });
    const statusData = await status.json();
    if (statusData.status === 'COMPLETED') {
      console.log(`  ✅ Complete!`);
      return statusData.outputUrl;
    }
    if (statusData.status === 'FAILED') {
      throw new Error('Sync Labs job failed');
    }
    process.stdout.write('.');
  }
  throw new Error('Sync Labs job timed out (10 min)');
}

async function main() {
  console.log('\n🔄 RETRY FAILED DAY 352 VIDEOS\n');
  console.log(`Retrying: ${FAILED_ARCHETYPES.join(', ')}\n`);

  const lesson = await getLesson();
  const script = `${lesson.headline}. ${lesson.fun_facts.join('. ')}. ${lesson.universal_truth}`;
  
  // Load existing manifest
  const manifestPath = 'generated-videos/sync-labs-redub/day-352-redub-manifest.json';
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
  
  let successCount = 0;
  
  for (const archetype of FAILED_ARCHETYPES) {
    console.log(`\n📹 ${archetype.toUpperCase()}`);
    try {
      const videoUrl = DAY_351_HEYGEN_URLS[archetype];
      if (!videoUrl) {
        console.log(`  ⚠️ No HeyGen URL for ${archetype}`);
        continue;
      }
      
      // Generate audio
      const audioBuffer = await generateAudio(script, `day352_${archetype}`);
      const filename = `day352_${archetype}_${Date.now()}.mp3`;
      const audioUrl = await uploadAudio(audioBuffer, filename);
      console.log(`  📤 Audio: ${audioUrl.substring(0, 60)}...`);
      
      // Redub
      const redubUrl = await redubWithSyncLabs(videoUrl, audioUrl);
      
      // Update manifest
      manifest.videos[archetype] = {
        archetype,
        success: true,
        referenceVideoUrl: videoUrl,
        newAudioUrl: audioUrl,
        redubVideoUrl: redubUrl,
        retriedAt: new Date().toISOString()
      };
      successCount++;
      
    } catch (error: any) {
      console.log(`  ❌ Error: ${error.message}`);
      manifest.videos[archetype] = {
        archetype,
        success: false,
        error: error.message,
        retriedAt: new Date().toISOString()
      };
    }
  }
  
  // Save updated manifest
  manifest.lastRetry = new Date().toISOString();
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  
  console.log(`\n✅ Retry complete: ${successCount}/${FAILED_ARCHETYPES.length} succeeded`);
  console.log(`📁 Updated: ${manifestPath}\n`);
}

main().catch(console.error);
