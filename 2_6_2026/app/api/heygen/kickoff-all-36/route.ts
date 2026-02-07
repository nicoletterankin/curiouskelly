/**
 * HEYGEN KICKOFF ALL 36 KELLY VARIANTS
 * 
 * Generates base videos for all 36 Kelly variants (3 ages x 12 archetypes).
 * These base videos can then be lip-synced with any audio using FAL.ai.
 * 
 * POST /api/heygen/kickoff-all-36
 * Body: { 
 *   test?: boolean,        // Use HeyGen test mode (no credits)
 *   startFrom?: string,    // Resume from specific key like "adult_explorer"
 *   limit?: number,        // Max videos to generate (default: 1 for safety)
 *   dryRun?: boolean       // Log payload without calling API
 * }
 * 
 * GET /api/heygen/kickoff-all-36
 * Returns current status of all 36 variants
 */

import { NextResponse } from 'next/server'
import { 
  KELLY_AVATAR_GROUPS, 
  ADULT_LOOKS, 
  KID_LOOKS, 
  SENIOR_LOOKS 
} from '@/lib/kelly-avatars'

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY

// Base script for all Kelly variants (short intro video ~15 seconds)
const BASE_SCRIPT = `Hi there! I'm Kelly, your curious learning companion. Every day, I have a brand new adventure waiting just for you. Ready to discover something amazing together?`

// All 12 archetypes
const ARCHETYPES = [
  'storyteller', 'explorer', 'scientist', 'architect',
  'strategist', 'diplomat', 'mystic', 'rebel',
  'macgyver', 'empath', 'provider', 'survivor'
] as const

type Archetype = typeof ARCHETYPES[number]

// Age categories
const AGES = ['kid', 'adult', 'senior'] as const
type AgeCategory = typeof AGES[number]

// Result type for tracking
interface GenerationResult {
  key: string
  age: AgeCategory
  archetype: Archetype
  avatarId: string
  lookId: string
  status: 'started' | 'skipped' | 'error' | 'dry_run'
  videoId?: string
  error?: string
  payload?: Record<string, unknown>
  response?: Record<string, unknown>
}

/**
 * Build the EXACT HeyGen API payload
 * This is the payload that will be sent to HeyGen support for verification
 */
function buildHeyGenPayload(
  avatarId: string, 
  lookId: string, 
  script: string,
  testMode: boolean
): Record<string, unknown> {
  return {
    video_inputs: [
      {
        character: {
          type: 'avatar',
          avatar_id: avatarId,
          look_id: lookId,
          avatar_style: 'normal'
        },
        voice: {
          type: 'text',
          input_text: script,
          voice_id: 'en-US-JennyNeural', // Azure TTS voice
          speed: 1.0
        },
        background: {
          type: 'color',
          value: '#FFFFFF'
        }
      }
    ],
    dimension: {
      width: 1080,
      height: 1920
    },
    aspect_ratio: '9:16',
    test: testMode
  }
}

/**
 * Get look ID for a given age and archetype
 */
function getLookId(age: AgeCategory, archetype: Archetype): string | null {
  const lookMaps = {
    kid: KID_LOOKS,
    adult: ADULT_LOOKS,
    senior: SENIOR_LOOKS
  }
  return lookMaps[age]?.[archetype] || null
}

export async function POST(request: Request) {
  // =========================================================================
  // VALIDATION
  // =========================================================================
  
  if (!HEYGEN_API_KEY) {
    console.error('[heygen/kickoff] HEYGEN_API_KEY not configured')
    return NextResponse.json({ 
      error: 'HEYGEN_API_KEY not configured',
      fix: 'Add HEYGEN_API_KEY to environment variables'
    }, { status: 500 })
  }

  // Parse request body
  const body = await request.json().catch(() => ({}))
  const testMode = body.test ?? true  // Default to test mode for safety
  const dryRun = body.dryRun ?? false
  const startFrom = body.startFrom as string | undefined
  const limit = body.limit ?? 1  // Default to 1 for safety (costs ~5 credits each)

  console.log('[heygen/kickoff] Starting with options:', { testMode, dryRun, startFrom, limit })

  // =========================================================================
  // GENERATE VIDEOS
  // =========================================================================
  
  const results: GenerationResult[] = []
  let started = !startFrom
  let count = 0

  for (const age of AGES) {
    const avatarId = KELLY_AVATAR_GROUPS[age]
    
    if (!avatarId) {
      console.error(`[heygen/kickoff] No avatar group ID for age: ${age}`)
      continue
    }

    for (const archetype of ARCHETYPES) {
      const key = `${age}_${archetype}`
      const lookId = getLookId(age, archetype)
      
      // Skip until we reach startFrom
      if (!started) {
        if (key === startFrom) {
          started = true
        } else {
          continue // Don't log skipped entries before startFrom
        }
      }

      // Check limit
      if (count >= limit) {
        results.push({ 
          key, age, archetype, 
          avatarId,
          lookId: lookId || 'MISSING',
          status: 'skipped',
          error: 'Limit reached'
        })
        continue
      }

      // Validate lookId exists
      if (!lookId) {
        results.push({ 
          key, age, archetype, 
          avatarId,
          lookId: 'MISSING',
          status: 'error',
          error: `No look_id found for ${key}. Check KELLY_AVATARS config.`
        })
        continue
      }

      // Build the exact payload
      const payload = buildHeyGenPayload(avatarId, lookId, BASE_SCRIPT, testMode)

      // =====================================================================
      // DRY RUN MODE - Just log the payload
      // =====================================================================
      if (dryRun) {
        results.push({
          key, age, archetype,
          avatarId,
          lookId,
          status: 'dry_run',
          payload
        })
        count++
        continue
      }

      // =====================================================================
      // ACTUAL API CALL
      // =====================================================================
      try {
        console.log(`[heygen/kickoff] Calling HeyGen API for ${key}`)
        console.log(`[heygen/kickoff] Payload:`, JSON.stringify(payload, null, 2))

        const response = await fetch('https://api.heygen.com/v2/video/generate', {
          method: 'POST',
          headers: {
            'X-Api-Key': HEYGEN_API_KEY,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(payload)
        })

        const responseText = await response.text()
        let responseData: Record<string, unknown> = {}
        
        try {
          responseData = JSON.parse(responseText)
        } catch {
          responseData = { raw: responseText }
        }

        console.log(`[heygen/kickoff] Response for ${key}:`, response.status, responseData)

        if (!response.ok) {
          results.push({ 
            key, age, archetype, 
            avatarId,
            lookId,
            status: 'error', 
            error: `HeyGen API error: ${response.status}`,
            payload,
            response: responseData
          })
          continue
        }

        const videoId = (responseData.data as Record<string, unknown>)?.video_id as string

        if (!videoId) {
          results.push({ 
            key, age, archetype, 
            avatarId,
            lookId,
            status: 'error', 
            error: 'No video_id in response',
            payload,
            response: responseData
          })
          continue
        }

        // Success!
        results.push({ 
          key, age, archetype, 
          avatarId,
          lookId,
          status: 'started', 
          videoId,
          payload,
          response: responseData
        })
        count++

        // Rate limiting: 500ms delay between requests
        await new Promise(r => setTimeout(r, 500))

      } catch (error) {
        console.error(`[heygen/kickoff] Exception for ${key}:`, error)
        results.push({ 
          key, age, archetype, 
          avatarId,
          lookId,
          status: 'error', 
          error: error instanceof Error ? error.message : 'Unknown error',
          payload
        })
      }
    }
  }

  // =========================================================================
  // SUMMARY
  // =========================================================================
  
  const summary = {
    timestamp: new Date().toISOString(),
    config: {
      testMode,
      dryRun,
      startFrom: startFrom || null,
      limit
    },
    stats: {
      total: 36,
      processed: results.length,
      started: results.filter(r => r.status === 'started').length,
      dryRun: results.filter(r => r.status === 'dry_run').length,
      skipped: results.filter(r => r.status === 'skipped').length,
      errors: results.filter(r => r.status === 'error').length
    },
    errors: results.filter(r => r.status === 'error').map(r => ({
      key: r.key,
      error: r.error,
      response: r.response
    })),
    results
  }

  console.log('[heygen/kickoff] Summary:', JSON.stringify(summary.stats))
  
  return NextResponse.json(summary)
}

/**
 * GET - Check status of all 36 variants
 */
export async function GET() {
  // Build the full matrix of 36 variants with their IDs
  const all36 = AGES.flatMap(age => 
    ARCHETYPES.map(archetype => {
      const key = `${age}_${archetype}`
      const avatarId = KELLY_AVATAR_GROUPS[age]
      const lookId = getLookId(age, archetype)
      
      return {
        key,
        age,
        archetype,
        avatarId: avatarId || 'MISSING',
        lookId: lookId || 'MISSING',
        valid: !!(avatarId && lookId)
      }
    })
  )

  const valid = all36.filter(v => v.valid).length
  const invalid = all36.filter(v => !v.valid).length

  return NextResponse.json({
    summary: {
      total: 36,
      valid,
      invalid,
      ready: valid === 36
    },
    avatarGroups: KELLY_AVATAR_GROUPS,
    variants: all36,
    samplePayload: buildHeyGenPayload(
      KELLY_AVATAR_GROUPS.adult,
      ADULT_LOOKS.storyteller,
      BASE_SCRIPT,
      true
    )
  })
}
