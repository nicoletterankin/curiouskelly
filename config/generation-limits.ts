/**
 * Daily generation limits and cost tracking configuration.
 * Engine STOPS and alerts if any limit is reached.
 */

export const DAILY_LIMITS = {
  // HeyGen credits per day (1 credit ≈ 1 minute of video)
  heygen_credits: 100,
  
  // ElevenLabs characters per day
  elevenlabs_characters: 500000,
  
  // Maximum USD spend per day across all services
  max_usd: 50,
  
  // Maximum retries for failed generations
  max_retries: 3,
  
  // Timeout for video generation (ms)
  generation_timeout_ms: 30 * 60 * 1000, // 30 minutes
};

export const COST_ESTIMATES = {
  // HeyGen: ~$0.20 per credit (approximate)
  heygen_per_credit_usd: 0.20,
  
  // ElevenLabs: ~$0.30 per 1000 characters
  elevenlabs_per_1k_chars_usd: 0.30,
  
  // OpenAI: ~$0.01 per 1000 tokens (GPT-4)
  openai_per_1k_tokens_usd: 0.01,
};

export const GENERATION_CONFIG = {
  // MVP archetype and age bucket
  mvp_archetype: 'The Explorer',
  mvp_age_bucket: 'adult',
  
  // Days to generate ahead
  lookahead_days: 3,
  
  // All phases in order
  phases: ['hook', 'cliff', 'q1', 'q2', 'q3', 'wisdom', 'outro'] as const,
  
  // Polling interval for HeyGen status
  heygen_poll_interval_ms: 30000, // 30 seconds
  
  // Bucket for video storage
  video_bucket: 'kelly-videos',
};

export const ARCHETYPES = [
  'The Explorer',
  'The Scientist', 
  'The Storyteller',
  'The Architect',
  'The Diplomat',
  'The Empath',
  'The MacGyver',
  'The Mystic',
  'The Provider',
  'The Rebel',
  'The Strategist',
  'The Survivor',
] as const;

export const AGE_BUCKETS = ['kid', 'teen', 'adult', 'mature', 'elder'] as const;

export type Archetype = typeof ARCHETYPES[number];
export type AgeBucket = typeof AGE_BUCKETS[number];
export type Phase = typeof GENERATION_CONFIG.phases[number];




