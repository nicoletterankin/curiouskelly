import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * =============================================================================
 * CURIOUS KELLY - DAY 20 VIDEO GENERATION QUEUE
 * Date: January 20, 2026
 * Lesson: "Why We Yawn"
 * =============================================================================
 * 
 * This endpoint queues video generation for all 5 phases of Day 20.
 * Two parallel pipelines: HeyGen (premium) and fal.ai (backup)
 * 
 * DOCUMENTATION FOR ANTIGRAVITY:
 * 
 * HeyGen Pipeline:
 * - Avatar ID: 3d6a9d6f91b444469dae87ebb3d9eba6 (Kelly Adult Storyteller)
 * - Voice ID: 1bd001e7e50f421d891986aad5158bc8 (ElevenLabs)
 * - Resolution: 1920x1080 (Apple Keynote quality)
 * - Status: BLOCKED - API key expired/invalid
 * 
 * fal.ai Pipeline:
 * - Model: sadtalker (lip-sync from image + audio)
 * - Image: Kelly Storyteller PNG from blob storage
 * - Status: READY
 * 
 * =============================================================================
 */

const DAY = 20
const TITLE = 'Why We Yawn'
const DATE = 'January 20, 2026'

// Kelly avatar configurations
const KELLY_CONFIG = {
  adult: {
    storyteller: {
      image_url: 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/kelly/archetypes/storyteller.png',
      heygen_avatar_id: '3d6a9d6f91b444469dae87ebb3d9eba6',
      voice_id: '1bd001e7e50f421d891986aad5158bc8'
    }
  }
}

// Full scripts for Day 20 - all 5 phases
// These are educator-quality scripts expanded from the lesson summaries
const SCRIPTS = {
  hook: `Have you noticed that yawns are contagious? When someone yawns, you might suddenly feel the urge to yawn too! Scientists think this happens because of mirror neurons in our brain that help us connect with others. But here's the really curious part - yawning isn't just about being tired. Your brain actually uses yawning to cool itself down, like a built-in air conditioner! Today we're going to explore why we yawn and what it tells us about how our amazing brains work.`,
  
  story: `Meet Maya, a curious 10-year-old who noticed something strange during her morning class. Her teacher yawned, and suddenly half the class started yawning too, including Maya! She wondered if yawning was like a secret signal spreading through the room. That night, Maya did an experiment - she fake-yawned in front of her dog, and guess what? Her dog yawned back! Maya discovered that yawning connects us to other beings, showing empathy and understanding without saying a word.`,
  
  wonder: `Here's what makes yawning truly fascinating: your brain heats up when you're thinking hard or feeling tired, and a big yawn brings cool air into your mouth and nose, which cools the blood going to your brain! It's like your body's natural reset button. Scientists have found that people yawn more in rooms that are warm and less in cool rooms. Even babies yawn in the womb before they're born! What do you think your yawns are telling you about how your brain is feeling?`,
  
  action: `Try this yawn experiment: Tomorrow, count how many times you yawn and write down what you were doing each time. Were you tired? Bored? Did someone else just yawn? See if you can spot patterns. Then try the "contagious yawn test" - yawn near friends and family and see how many catch it. People who catch yawns easily often score higher on empathy tests! Notice if reading this made you yawn - that counts too!`,
  
  wisdom: `Yawning teaches us that our bodies are constantly communicating, even without words. When you catch a yawn from someone, your brain is saying "I understand you, I connect with you." This simple act shows how deeply we're wired to bond with others. Your body has its own wisdom - it knows when to cool your brain, when to take a deep breath, and when to show others that you're paying attention. Trust your body's signals, they're smarter than you might think!`
}

// Phase metadata
const PHASE_ORDER = ['hook', 'story', 'wonder', 'action', 'wisdom'] as const
type Phase = typeof PHASE_ORDER[number]

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const action = searchParams.get('action')
  
  // Check environment variables
  const heygenKey = process.env.HEYGEN_API_KEY
  const falKey = process.env.FAL_KEY
  const elevenLabsKey = process.env.ELEVENLABS_API_KEY
  
  const config = KELLY_CONFIG.adult.storyteller

  // =========================================================================
  // ACTION: status - Show current pipeline status
  // =========================================================================
  if (!action || action === 'status') {
    return NextResponse.json({
      '=== CURIOUS KELLY DAY 20 VIDEO QUEUE ===': '',
      day: DAY,
      title: TITLE,
      date: DATE,
      
      pipelines: {
        heygen: {
          status: heygenKey ? 'API_KEY_SET' : 'MISSING_KEY',
          last_error: '401 Unauthorized - key expired/invalid',
          action_needed: 'Update HEYGEN_API_KEY in Vercel dashboard',
          avatar_id: config.heygen_avatar_id,
          voice_id: config.voice_id,
          resolution: '1920x1080'
        },
        fal_ai: {
          status: falKey ? 'READY' : 'MISSING_KEY',
          model: 'sadtalker',
          image_url: config.image_url,
          note: 'Requires audio file - will use ElevenLabs first'
        },
        elevenlabs: {
          status: elevenLabsKey ? 'READY' : 'MISSING_KEY',
          voice_id: config.voice_id
        }
      },
      
      phases: PHASE_ORDER.map(p => ({
        phase: p,
        script_length: SCRIPTS[p].length,
        script_preview: SCRIPTS[p].substring(0, 100) + '...'
      })),
      
      actions: {
        queue_heygen: '/api/video-queue/day20?action=queue-heygen',
        queue_fal: '/api/video-queue/day20?action=queue-fal',
        queue_both: '/api/video-queue/day20?action=queue-all',
        generate_audio: '/api/video-queue/day20?action=generate-audio&phase=hook',
        full_payload: '/api/video-queue/day20?action=show-payload'
      }
    })
  }

  // =========================================================================
  // ACTION: show-payload - Show exact API payloads for both pipelines
  // =========================================================================
  if (action === 'show-payload') {
    return NextResponse.json({
      '=== EXACT API PAYLOADS FOR DAY 20 ===': '',
      
      heygen_payload: {
        endpoint: 'POST https://api.heygen.com/v2/video/generate',
        headers: {
          'X-Api-Key': 'HEYGEN_API_KEY',
          'Content-Type': 'application/json'
        },
        body: {
          video_inputs: [{
            character: {
              type: 'talking_photo',
              talking_photo_id: config.heygen_avatar_id
            },
            voice: {
              type: 'text',
              input_text: '<<SCRIPT_FOR_PHASE>>',
              voice_id: config.voice_id
            }
          }],
          dimension: { width: 1920, height: 1080 },
          test: false
        }
      },
      
      fal_payload: {
        endpoint: 'POST https://queue.fal.run/fal-ai/sadtalker',
        headers: {
          'Authorization': 'Key FAL_KEY',
          'Content-Type': 'application/json'
        },
        body: {
          source_image_url: config.image_url,
          driven_audio_url: '<<ELEVENLABS_AUDIO_URL>>',
          preprocess: 'crop',
          still_mode: false,
          expression_scale: 1.2
        }
      },
      
      elevenlabs_payload: {
        endpoint: 'POST https://api.elevenlabs.io/v1/text-to-speech/' + config.voice_id,
        headers: {
          'xi-api-key': 'ELEVENLABS_API_KEY',
          'Content-Type': 'application/json'
        },
        body: {
          text: '<<SCRIPT_FOR_PHASE>>',
          model_id: 'eleven_multilingual_v2',
          voice_settings: {
            stability: 0.5,
            similarity_boost: 0.75
          }
        }
      },
      
      all_scripts: SCRIPTS
    })
  }

  // =========================================================================
  // ACTION: generate-audio - Generate audio for a single phase via ElevenLabs
  // =========================================================================
  if (action === 'generate-audio') {
    const phase = (searchParams.get('phase') || 'hook') as Phase
    const script = SCRIPTS[phase]
    
    if (!elevenLabsKey) {
      return NextResponse.json({ error: 'ELEVENLABS_API_KEY not configured' }, { status: 500 })
    }
    
    const res = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${config.voice_id}`, {
      method: 'POST',
      headers: {
        'xi-api-key': elevenLabsKey,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        text: script,
        model_id: 'eleven_multilingual_v2',
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.75
        }
      })
    })
    
    if (!res.ok) {
      const err = await res.text()
      return NextResponse.json({ error: 'ElevenLabs error', status: res.status, details: err }, { status: res.status })
    }
    
    // Return audio as base64 or save to blob
    const audioBuffer = await res.arrayBuffer()
    
    return new NextResponse(audioBuffer, {
      headers: {
        'Content-Type': 'audio/mpeg',
        'Content-Disposition': `attachment; filename="day20-${phase}-adult-storyteller.mp3"`
      }
    })
  }

  // =========================================================================
  // ACTION: queue-heygen - Queue all phases to HeyGen
  // =========================================================================
  if (action === 'queue-heygen') {
    if (!heygenKey) {
      return NextResponse.json({ 
        error: 'HEYGEN_API_KEY not configured',
        fix: 'Add valid API key to Vercel environment variables'
      }, { status: 500 })
    }
    
    const results = []
    
    for (const phase of PHASE_ORDER) {
      const payload = {
        video_inputs: [{
          character: {
            type: 'talking_photo',
            talking_photo_id: config.heygen_avatar_id
          },
          voice: {
            type: 'text',
            input_text: SCRIPTS[phase],
            voice_id: config.voice_id
          }
        }],
        dimension: { width: 1920, height: 1080 },
        test: false
      }
      
      try {
        const res = await fetch('https://api.heygen.com/v2/video/generate', {
          method: 'POST',
          headers: {
            'X-Api-Key': heygenKey,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(payload)
        })
        
        const data = await res.json()
        
        results.push({
          phase,
          status: res.ok ? 'queued' : 'failed',
          video_id: data.data?.video_id,
          error: res.ok ? null : data.error
        })
        
        // Save to DB if successful
        if (res.ok && data.data?.video_id) {
          await sql`
            INSERT INTO heygen_videos (day_of_year, phase, age_category, archetype, status, heygen_video_id, script, created_at)
            VALUES (${DAY}, ${phase}, 'adult', 'storyteller', 'pending', ${data.data.video_id}, ${SCRIPTS[phase]}, NOW())
            ON CONFLICT (day_of_year, phase, age_category, archetype) DO UPDATE SET
              heygen_video_id = ${data.data.video_id},
              status = 'pending',
              script = ${SCRIPTS[phase]}
          `.catch((err) => console.error('DB insert failed:', err))
        }
      } catch (err) {
        results.push({ phase, status: 'error', error: String(err) })
      }
    }
    
    return NextResponse.json({
      pipeline: 'heygen',
      day: DAY,
      title: TITLE,
      queued: results.filter(r => r.status === 'queued').length,
      failed: results.filter(r => r.status !== 'queued').length,
      results
    })
  }

  // =========================================================================
  // ACTION: queue-fal - Queue all phases to fal.ai (requires audio first)
  // =========================================================================
  if (action === 'queue-fal') {
    if (!falKey) {
      return NextResponse.json({ error: 'FAL_KEY not configured' }, { status: 500 })
    }
    
    return NextResponse.json({
      pipeline: 'fal',
      status: 'requires_audio',
      message: 'fal.ai SadTalker requires audio files. Generate audio first.',
      steps: [
        '1. Generate audio: /api/video-queue/day20?action=generate-audio&phase=hook',
        '2. Upload audio to blob storage',
        '3. Call fal with audio URL: /api/fal/day20?action=generate&phase=hook&audio_url=...'
      ],
      alternative: 'Use HeyGen which handles TTS internally (when API key is valid)'
    })
  }

  return NextResponse.json({ error: 'Unknown action', valid_actions: ['status', 'show-payload', 'generate-audio', 'queue-heygen', 'queue-fal'] })
}
