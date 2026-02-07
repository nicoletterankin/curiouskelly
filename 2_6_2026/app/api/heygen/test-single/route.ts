/**
 * HEYGEN SINGLE TEST - Generate ONE video with full logging
 * 
 * This endpoint generates a single Kelly video and logs EVERYTHING.
 * Use this to verify the payload format before running all 36.
 * 
 * POST /api/heygen/test-single
 * Body: {
 *   age?: 'kid' | 'adult' | 'senior',  // default: 'adult'
 *   archetype?: string,                 // default: 'storyteller'
 *   test?: boolean,                     // default: true (no credits)
 *   dryRun?: boolean                    // default: false
 * }
 * 
 * Response includes:
 * - exactPayload: The exact JSON sent to HeyGen
 * - curlCommand: Copy-paste curl for manual testing
 * - response: Full HeyGen API response
 */

import { NextResponse } from 'next/server'
import { 
  KELLY_AVATAR_GROUPS, 
  ADULT_LOOKS, 
  KID_LOOKS, 
  SENIOR_LOOKS 
} from '@/lib/kelly-avatars'

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY

// Short test script (~10 seconds of speech)
const TEST_SCRIPT = `Hi there! I'm Kelly, your curious learning companion. Ready to discover something amazing together?`

type AgeCategory = 'kid' | 'adult' | 'senior'
type Archetype = keyof typeof ADULT_LOOKS

function getLookId(age: AgeCategory, archetype: Archetype): string | null {
  const lookMaps = {
    kid: KID_LOOKS,
    adult: ADULT_LOOKS,
    senior: SENIOR_LOOKS
  }
  return (lookMaps[age] as Record<string, string>)?.[archetype] || null
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => ({}))
  
  const age: AgeCategory = body.age || 'adult'
  const archetype: Archetype = body.archetype || 'storyteller'
  const testMode = body.test ?? true
  const dryRun = body.dryRun ?? false

  // Get IDs
  const avatarId = KELLY_AVATAR_GROUPS[age]
  const lookId = getLookId(age, archetype)

  if (!avatarId || !lookId) {
    return NextResponse.json({
      error: 'Invalid configuration',
      details: {
        age,
        archetype,
        avatarId: avatarId || 'MISSING',
        lookId: lookId || 'MISSING'
      }
    }, { status: 400 })
  }

  // Build EXACT payload
  const exactPayload = {
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
          input_text: TEST_SCRIPT,
          voice_id: 'en-US-JennyNeural',
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

  // Generate curl command for manual testing
  const curlCommand = `curl -X POST "https://api.heygen.com/v2/video/generate" \\
  -H "X-Api-Key: YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '${JSON.stringify(exactPayload)}'`

  // Log everything
  console.log('========================================')
  console.log('[heygen/test-single] Configuration:')
  console.log('  Age:', age)
  console.log('  Archetype:', archetype)
  console.log('  Avatar ID:', avatarId)
  console.log('  Look ID:', lookId)
  console.log('  Test Mode:', testMode)
  console.log('  Dry Run:', dryRun)
  console.log('----------------------------------------')
  console.log('[heygen/test-single] Exact Payload:')
  console.log(JSON.stringify(exactPayload, null, 2))
  console.log('========================================')

  // If dry run, return without calling API
  if (dryRun) {
    return NextResponse.json({
      mode: 'DRY_RUN',
      message: 'Payload generated but API not called',
      config: { age, archetype, avatarId, lookId, testMode },
      exactPayload,
      curlCommand
    })
  }

  // Verify API key
  if (!HEYGEN_API_KEY) {
    return NextResponse.json({
      error: 'HEYGEN_API_KEY not configured',
      config: { age, archetype, avatarId, lookId },
      exactPayload,
      curlCommand
    }, { status: 500 })
  }

  // Call HeyGen API
  try {
    console.log('[heygen/test-single] Calling HeyGen API...')
    
    const response = await fetch('https://api.heygen.com/v2/video/generate', {
      method: 'POST',
      headers: {
        'X-Api-Key': HEYGEN_API_KEY,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(exactPayload)
    })

    const responseText = await response.text()
    console.log('[heygen/test-single] Response Status:', response.status)
    console.log('[heygen/test-single] Response Body:', responseText)

    let responseData: Record<string, unknown>
    try {
      responseData = JSON.parse(responseText)
    } catch {
      responseData = { rawText: responseText }
    }

    if (!response.ok) {
      return NextResponse.json({
        error: 'HeyGen API Error',
        status: response.status,
        config: { age, archetype, avatarId, lookId, testMode },
        exactPayload,
        curlCommand,
        response: responseData,
        troubleshooting: {
          checkAvatarId: `Verify avatar_id ${avatarId} exists in your HeyGen account`,
          checkLookId: `Verify look_id ${lookId} exists for this avatar`,
          checkApiKey: 'Verify HEYGEN_API_KEY is valid and has credits',
          documentation: 'https://docs.heygen.com/docs/create-video'
        }
      }, { status: response.status })
    }

    const videoId = (responseData.data as Record<string, unknown>)?.video_id

    return NextResponse.json({
      success: true,
      message: testMode 
        ? 'Test video generation started (no credits used)' 
        : 'Video generation started',
      config: { age, archetype, avatarId, lookId, testMode },
      videoId,
      exactPayload,
      curlCommand,
      response: responseData,
      nextSteps: {
        checkStatus: `/api/heygen/status?videoId=${videoId}`,
        note: 'Video typically takes 2-5 minutes to complete'
      }
    })

  } catch (error) {
    console.error('[heygen/test-single] Exception:', error)
    return NextResponse.json({
      error: 'Request failed',
      message: error instanceof Error ? error.message : 'Unknown error',
      config: { age, archetype, avatarId, lookId },
      exactPayload,
      curlCommand
    }, { status: 500 })
  }
}

export async function GET() {
  // Return sample payloads for all 36 variants
  const samples = (['adult', 'kid', 'senior'] as const).map(age => ({
    age,
    avatarId: KELLY_AVATAR_GROUPS[age],
    archetypes: Object.entries(age === 'kid' ? KID_LOOKS : age === 'senior' ? SENIOR_LOOKS : ADULT_LOOKS)
      .map(([archetype, lookId]) => ({ archetype, lookId }))
  }))

  return NextResponse.json({
    endpoint: 'POST /api/heygen/test-single',
    description: 'Test single Kelly video generation',
    testModeDefault: true,
    samples,
    exampleBody: {
      age: 'adult',
      archetype: 'storyteller',
      test: true,
      dryRun: false
    }
  })
}
