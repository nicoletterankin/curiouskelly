// ============================================================================
// HEYGEN API CLIENT
// Handles video generation, status checking, and avatar management
// ============================================================================

import {
  HEYGEN_AVATAR_GROUPS,
  HEYGEN_KID_KELLY_LOOKS,
  HEYGEN_ADULT_KELLY_LOOKS,
  HEYGEN_SENIOR_KELLY_LOOKS,
  getAgeCategory,
} from './kelly-assets'

// ============================================================================
// TYPES
// ============================================================================

export type AgeCategory = 'kid' | 'adult' | 'senior'
export type Archetype = 
  | 'scientist' | 'explorer' | 'rebel' | 'architect'
  | 'diplomat' | 'empath' | 'macgyver' | 'mystic'
  | 'provider' | 'storyteller' | 'strategist' | 'survivor'

export interface HeyGenVideoRequest {
  avatarGroupId: string
  lookId: string
  script: string
  voiceId?: string
  voiceUrl?: string // For ElevenLabs audio
  title?: string
  test?: boolean
}

export interface HeyGenVideoResponse {
  video_id: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
}

export interface HeyGenVideoStatus {
  video_id: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  video_url?: string
  thumbnail_url?: string
  duration?: number
  error?: string
}

export interface HeyGenAvatar {
  avatar_id: string
  avatar_name: string
  looks: {
    look_id: string
    look_name: string
    thumbnail_url: string
  }[]
}

// ============================================================================
// CONFIGURATION
// ============================================================================

const HEYGEN_API_BASE = 'https://api.heygen.com/v2'

// Read API key at runtime to pick up env var changes
function getApiKey(): string | undefined {
  return process.env.HEYGEN_API_KEY
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get avatar configuration for a given age and archetype
 */
export function getAvatarConfig(age: number | AgeCategory, archetype: Archetype) {
  const ageCategory = typeof age === 'number' ? getAgeCategory(age) : age
  
  const avatarGroupId = HEYGEN_AVATAR_GROUPS[ageCategory]
  
  const lookMaps = {
    kid: HEYGEN_KID_KELLY_LOOKS,
    adult: HEYGEN_ADULT_KELLY_LOOKS,
    senior: HEYGEN_SENIOR_KELLY_LOOKS,
  }
  
  const lookId = lookMaps[ageCategory]?.[archetype]
  
  if (!avatarGroupId || !lookId) {
    throw new Error(`Invalid avatar config: age=${ageCategory}, archetype=${archetype}`)
  }
  
  return {
    avatarGroupId,
    lookId,
    ageCategory,
    archetype,
    avatarName: `kelly_${ageCategory}_${archetype}`,
  }
}

/**
 * Parse avatar name into components
 */
export function parseAvatarName(name: string): { age: AgeCategory; archetype: Archetype } | null {
  const match = name.match(/^kelly_(kid|adult|senior)_(\w+)$/)
  if (!match) return null
  return {
    age: match[1] as AgeCategory,
    archetype: match[2] as Archetype,
  }
}

// ============================================================================
// API CLIENT
// ============================================================================

class HeyGenClient {
  private baseUrl: string

  constructor() {
    this.baseUrl = HEYGEN_API_BASE
  }
  
  private get apiKey(): string {
    return getApiKey() || ''
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const key = this.apiKey
    if (!key) {
      throw new Error('HeyGen API key not configured - set HEYGEN_API_KEY env var')
    }
    
    // Log key prefix for debugging (don't log full key)
    console.log(`[HeyGen] Making request to ${endpoint} with key starting: ${key.substring(0, 10)}...`)

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'X-Api-Key': key,
        'Content-Type': 'application/json',
        ...options.headers,
      },
    })

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`HeyGen API error: ${response.status} - ${error}`)
    }

    return response.json()
  }

  /**
   * Generate a video with the specified avatar and script
   * IMPORTANT: Kelly avatars are PHOTO avatars, use talking_photo_id (the lookId)
   */
  async generateVideo(request: HeyGenVideoRequest): Promise<HeyGenVideoResponse> {
    // For Kelly photo avatars, we use talking_photo with the lookId directly
    // The lookId IS the talking_photo_id for each archetype variant
    const payload: Record<string, unknown> = {
      video_inputs: [
        {
          character: {
            type: 'talking_photo',
            talking_photo_id: request.lookId, // Use lookId as the talking_photo_id
          },
          voice: request.voiceUrl
            ? {
                type: 'audio',
                audio_url: request.voiceUrl,
              }
            : {
                type: 'text',
                input_text: request.script,
                voice_id: request.voiceId || 'default',
              },
        },
      ],
      dimension: {
        width: 1920,
        height: 1080,
      },
      test: request.test ?? false,
    }

    if (request.title) {
      payload.title = request.title
    }

    const response = await this.request<{ data: HeyGenVideoResponse }>(
      '/video/generate',
      {
        method: 'POST',
        body: JSON.stringify(payload),
      }
    )

    return response.data
  }

  /**
   * Generate video using pre-recorded audio (ElevenLabs)
   */
  async generateVideoWithAudio(
    avatarGroupId: string,
    lookId: string,
    audioUrl: string,
    title?: string
  ): Promise<HeyGenVideoResponse> {
    return this.generateVideo({
      avatarGroupId,
      lookId,
      script: '', // Not used when audio provided
      voiceUrl: audioUrl,
      title,
    })
  }

  /**
   * Check the status of a video generation request
   */
  async getVideoStatus(videoId: string): Promise<HeyGenVideoStatus> {
    const response = await this.request<{ data: HeyGenVideoStatus }>(
      `/video_status.get?video_id=${videoId}`,
      { method: 'GET' }
    )
    return response.data
  }

  /**
   * Poll for video completion
   */
  async waitForVideo(
    videoId: string,
    options: { maxWaitMs?: number; pollIntervalMs?: number } = {}
  ): Promise<HeyGenVideoStatus> {
    const { maxWaitMs = 300000, pollIntervalMs = 5000 } = options
    const startTime = Date.now()

    while (Date.now() - startTime < maxWaitMs) {
      const status = await this.getVideoStatus(videoId)
      
      if (status.status === 'completed') {
        return status
      }
      
      if (status.status === 'failed') {
        throw new Error(`Video generation failed: ${status.error}`)
      }

      await new Promise(resolve => setTimeout(resolve, pollIntervalMs))
    }

    throw new Error('Video generation timed out')
  }

  /**
   * List available avatars
   */
  async listAvatars(): Promise<HeyGenAvatar[]> {
    const response = await this.request<{ data: { avatars: HeyGenAvatar[] } }>(
      '/avatars',
      { method: 'GET' }
    )
    return response.data.avatars
  }

  /**
   * Get remaining credits
   */
  async getCredits(): Promise<{ remaining: number; used: number }> {
    const response = await this.request<{ data: { remaining_quota: number; used_quota: number } }>(
      '/user/remaining_quota',
      { method: 'GET' }
    )
    return {
      remaining: response.data.remaining_quota,
      used: response.data.used_quota,
    }
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

// Singleton instance
export const heygenClient = new HeyGenClient()

// Export class for custom instances
export { HeyGenClient }

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/**
 * Generate a Kelly video for a specific lesson phase
 */
export async function generateKellyVideo(params: {
  age: number | AgeCategory
  archetype: Archetype
  script: string
  audioUrl?: string
  title?: string
  test?: boolean
}): Promise<HeyGenVideoResponse> {
  const config = getAvatarConfig(params.age, params.archetype)
  
  if (params.audioUrl) {
    return heygenClient.generateVideoWithAudio(
      config.avatarGroupId,
      config.lookId,
      params.audioUrl,
      params.title
    )
  }
  
  return heygenClient.generateVideo({
    avatarGroupId: config.avatarGroupId,
    lookId: config.lookId,
    script: params.script,
    title: params.title,
    test: params.test,
  })
}

/**
 * Generate all 5 phase videos for a lesson
 */
export async function generateLessonVideos(params: {
  dayOfYear: number
  age: number | AgeCategory
  archetype: Archetype
  scripts: {
    hook: string
    story: string
    wonder: string
    action: string
    wisdom: string
  }
  audioUrls?: {
    hook?: string
    story?: string
    wonder?: string
    action?: string
    wisdom?: string
  }
}): Promise<Record<string, HeyGenVideoResponse>> {
  const phases = ['hook', 'story', 'wonder', 'action', 'wisdom'] as const
  const results: Record<string, HeyGenVideoResponse> = {}
  
  for (const phase of phases) {
    const title = `Day ${params.dayOfYear} - ${phase.toUpperCase()}`
    
    results[phase] = await generateKellyVideo({
      age: params.age,
      archetype: params.archetype,
      script: params.scripts[phase],
      audioUrl: params.audioUrls?.[phase],
      title,
    })
    
    // Small delay to avoid rate limiting
    await new Promise(resolve => setTimeout(resolve, 500))
  }
  
  return results
}

/**
 * Get avatar ID (lookId) for a given age category and archetype
 */
export function getAvatarId(ageCategory: AgeCategory, archetype: Archetype): string | null {
  try {
    const config = getAvatarConfig(ageCategory, archetype)
    return config.lookId
  } catch {
    return null
  }
}

/**
 * Get all 36 avatar configurations
 */
export function getAllAvatarConfigs() {
  const ages: AgeCategory[] = ['kid', 'adult', 'senior']
  const archetypes: Archetype[] = [
    'scientist', 'explorer', 'rebel', 'architect',
    'diplomat', 'empath', 'macgyver', 'mystic',
    'provider', 'storyteller', 'strategist', 'survivor',
  ]
  
  const configs: ReturnType<typeof getAvatarConfig>[] = []
  
  for (const age of ages) {
    for (const archetype of archetypes) {
      configs.push(getAvatarConfig(age, archetype))
    }
  }
  
  return configs
}
