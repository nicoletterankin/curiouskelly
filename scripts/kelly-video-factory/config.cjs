/**
 * Kelly Video Factory - Configuration
 * 
 * Central configuration for all Kelly video generation.
 * Tuned for quality, speed, and reliability.
 */

module.exports = {
  // Kelly LoRA configuration
  lora: {
    weights: 'huggingface.co/CuriousKellycom/curious-kelly-lora',
    scale: 0.85,
    triggerWord: 'kelly',
  },
  
  // Character specification (from production factory)
  character: {
    hair: 'long wavy chestnut brown hair with subtle highlights',
    eyes: 'warm brown eyes with visible catchlights',
    outfit: 'soft powder blue crewneck sweater',
    skinTone: 'natural warm skin tone',
    age: 'early 30s',
    // Core identity traits for consistency
    identity: 'friendly approachable teacher, intelligent warmth, genuine smile lines, natural beauty',
    // Style keywords for cinematic quality
    style: 'cinematic lighting, shallow depth of field, 85mm lens, professional color grading, soft diffused lighting',
    // Negative prompts to avoid common issues
    negativePrompt: 'pink sweater, red sweater, beige sweater, teal sweater, green sweater, yellow sweater, deformed, blurry, bad anatomy, extra fingers, mutated hands, poorly drawn face, mutation, disfigured, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, out of frame, cropped, watermark, signature, text',
  },
  
  // Quality tiers
  // Note: Flux Dev LoRA only supports megapixels: "1" or "0.25"
  quality: {
    preview: {
      name: 'Preview',
      imageSize: '1:1',
      imageMegapixels: '0.25', // Small for speed
      animationModel: 'svd',
      animationFrames: 14,
      lipsyncModel: 'wav2lip',
      upscale: false,
      estimatedTime: 90, // seconds
    },
    standard: {
      name: 'Standard',
      imageSize: '16:9',
      imageMegapixels: '1', // Full resolution
      animationModel: 'svd',
      animationFrames: 25,
      lipsyncModel: 'wav2lip',
      upscale: false,
      estimatedTime: 180,
    },
    production: {
      name: 'Production',
      imageSize: '16:9',
      imageMegapixels: '1',
      animationModel: 'svd_xt',
      animationFrames: 25,
      lipsyncModel: 'sadtalker_hq',
      upscale: true, // 4K upscale
      estimatedTime: 300,
    },
  },
  
  // Replicate models
  models: {
    imageGeneration: {
      id: 'black-forest-labs/flux-dev-lora',
      version: null, // Fetched at runtime
    },
    animation: {
      svd: {
        id: 'stability-ai/stable-video-diffusion',
        version: '3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438',
        videoLength: '14_frames_with_svd',
      },
      svd_xt: {
        id: 'stability-ai/stable-video-diffusion',
        version: '3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438',
        videoLength: '25_frames_with_svd_xt',
      },
    },
    lipsync: {
      wav2lip: {
        id: 'devxpy/wav2lip',
        version: '8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef',
        estimatedTime: 10,
      },
      sadtalker_hq: {
        id: 'cjwbw/sadtalker',
        version: 'a519cc0cfebaaeade068b23899165a11ec76aaa1d2b313d40d214f204ec957a3',
        estimatedTime: 60,
        options: {
          enhancer: 'gfpgan',
          preprocess: 'crop',
          still_mode: false,
        },
      },
    },
    upscaler: {
      id: 'lucataco/real-esrgan-video',
      version: 'c23768236472c41b7a121ee735c8073e29080c02d343419c4b7f0e56e045cb4d',
      scale: 4, // 4x upscale
    },
  },
  
  // ElevenLabs voice
  voice: {
    id: process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0',
    model: 'eleven_turbo_v2_5',
    settings: {
      stability: 0.5,
      similarity_boost: 0.85,
      use_speaker_boost: true,
    },
  },
  
  // Storage
  storage: {
    bucket: 'kelly-templates',
    paths: {
      images: 'production/images',
      animations: 'production/animations', 
      videos: 'production/videos',
      audio: 'production/audio',
    },
  },
  
  // Retry configuration
  retry: {
    maxAttempts: 3,
    delayMs: 5000,
    backoffMultiplier: 2,
  },
  
  // Polling configuration
  polling: {
    intervalMs: 3000,
    maxAttempts: 200, // 10 minutes max
  },
  
  // Template prompts - ENHANCED for production quality
  // Each prompt uses: {triggerWord}, {hair}, {eyes}, {outfit}, {style}
  templates: {
    // HOOK phase - Grab attention, create wonder
    excited: {
      prompt: '{triggerWord}, {identity}, woman with {hair} and {eyes}, wearing {outfit}, medium close-up shot, eyes sparkling with genuine excitement and wonder, natural joyful expression with teeth showing, hands gesturing expressively mid-explanation, warm modern classroom environment with soft bokeh background, golden hour window light mixing with ambient lighting, {style}, capturing a moment of pure discovery and enthusiasm',
      environment: 'classroom',
      emotion: 'wonder',
      action: 'expressive_gesture',
      cameraAngle: 'eye_level',
      framing: 'medium_closeup',
    },
    
    // Q1 phase - Curiosity, inviting exploration
    curious: {
      prompt: '{triggerWord}, {identity}, woman with {hair} and {eyes}, wearing {outfit}, head slightly tilted with one eyebrow raised in genuine curiosity, warm inviting smile, one hand gesturing as if asking a question, cozy study room with warm wood tones and soft lamplight, books subtly visible in background, {style}, intimate teacher-student moment feel',
      environment: 'study',
      emotion: 'curious_inviting',
      action: 'questioning_gesture',
      cameraAngle: 'slight_high',
      framing: 'medium',
    },
    
    // Q2 phase - Teaching, explaining with passion
    explain: {
      prompt: '{triggerWord}, {identity}, woman with {hair} and {eyes}, wearing {outfit}, animated expression while explaining something fascinating, hands positioned as if holding an invisible concept, leaning slightly forward with engaged posture, professional studio setting with soft rim lighting and dark teal gradient background, {style}, the energy of someone who loves sharing knowledge',
      environment: 'studio',
      emotion: 'engaged_passionate',
      action: 'concept_gesture',
      cameraAngle: 'eye_level',
      framing: 'medium_wide',
    },
    
    // Q3 phase - Deeper thinking, reflection
    thoughtful: {
      prompt: '{triggerWord}, {identity}, woman with {hair} and {eyes}, wearing {outfit}, contemplative expression with a soft knowing smile, chin resting gently on hand, gazing slightly off-camera as if pondering something profound, warm library setting with leather-bound books and soft golden afternoon light streaming through windows, {style}, wisdom and reflection',
      environment: 'library',
      emotion: 'contemplative',
      action: 'thinking_pose',
      cameraAngle: 'three_quarter',
      framing: 'close_up',
    },
    
    // WISDOM phase - Heartfelt connection, emotional landing
    heartfelt: {
      prompt: '{triggerWord}, {identity}, woman with {hair} and {eyes}, wearing {outfit}, close-up portrait, hand placed gently over heart, eyes filled with genuine warmth and sincerity looking directly at camera, soft empathetic smile, warm cozy environment with soft diffused golden backlighting creating a gentle glow, {style}, intimate moment of genuine human connection and care',
      environment: 'warm_intimate',
      emotion: 'sincere_caring',
      action: 'heart_touch',
      cameraAngle: 'eye_level_intimate',
      framing: 'close_up_portrait',
    },
    
    // WELCOME - Opening, greeting (for special intros)
    welcome: {
      prompt: '{triggerWord}, {identity}, woman with {hair} and {eyes}, wearing {outfit}, full body shot, arms open in welcoming gesture, genuine warm smile, standing in beautiful sunlit nature setting with soft morning light, trees with golden leaves in background, {style}, the feeling of being warmly invited into something wonderful',
      environment: 'nature',
      emotion: 'welcoming',
      action: 'open_arms',
      cameraAngle: 'slight_low',
      framing: 'full_body',
    },
    
    // CELEBRATING - Success moment (for correct answers)
    celebrating: {
      prompt: '{triggerWord}, {identity}, woman with {hair} and {eyes}, wearing {outfit}, genuinely proud and delighted expression, subtle clapping or thumbs up gesture, eyes crinkled with authentic happiness, warm celebratory lighting with subtle confetti-like bokeh in background, {style}, the joy of watching someone succeed',
      environment: 'celebration',
      emotion: 'proud_delighted',
      action: 'celebration',
      cameraAngle: 'eye_level',
      framing: 'medium',
    },
    
    // ENCOURAGING - Supportive (for redirecting wrong answers)
    encouraging: {
      prompt: '{triggerWord}, {identity}, woman with {hair} and {eyes}, wearing {outfit}, gentle encouraging expression with a reassuring smile, head tilted with empathy, one hand raised in a supportive gesture, soft warm lighting with calming blue-green tones in background, {style}, the patience and kindness of a supportive mentor',
      environment: 'supportive',
      emotion: 'encouraging',
      action: 'supportive_gesture',
      cameraAngle: 'eye_level',
      framing: 'medium_closeup',
    },
    
    // LISTENING - Attentive (waiting for user response)
    listening: {
      prompt: '{triggerWord}, {identity}, woman with {hair} and {eyes}, wearing {outfit}, attentive listening expression with head slightly tilted, warm patient smile, hands clasped comfortably, soft natural lighting with neutral calming background, {style}, fully present and genuinely interested in what you have to say',
      environment: 'neutral',
      emotion: 'attentive',
      action: 'listening',
      cameraAngle: 'eye_level',
      framing: 'medium',
    },
  },
};

