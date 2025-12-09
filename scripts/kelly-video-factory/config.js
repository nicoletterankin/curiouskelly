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
    hair: 'long wavy brown hair',
    eyes: 'brown eyes',
    outfit: 'powder blue sweater',
    // Negative prompts to avoid common issues
    negativePrompt: 'pink sweater, red sweater, beige sweater, teal sweater, deformed, blurry, bad anatomy',
  },
  
  // Quality tiers
  quality: {
    preview: {
      name: 'Preview',
      imageSize: '1:1',
      imageMegapixels: '0.5',
      animationModel: 'svd',
      animationFrames: 14,
      lipsyncModel: 'wav2lip',
      upscale: false,
      estimatedTime: 90, // seconds
    },
    standard: {
      name: 'Standard',
      imageSize: '16:9',
      imageMegapixels: '1',
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
  
  // Template prompts
  templates: {
    welcome: {
      prompt: '{triggerWord}, woman with {hair} and {eyes}, wearing {outfit}, standing on sunlit forest path, arms open in welcoming gesture, warm genuine smile, full body shot, professional photography, 4K',
      environment: 'forest',
      emotion: 'warm',
      action: 'arms_open',
    },
    explain: {
      prompt: '{triggerWord}, woman with {hair} and {eyes}, wearing {outfit}, sitting in directors chair in studio with dark background, natural hand gestures while explaining, engaged expression, professional lighting, 4K',
      environment: 'studio',
      emotion: 'engaged',
      action: 'gesturing',
    },
    heartfelt: {
      prompt: '{triggerWord}, woman with {hair} and {eyes}, wearing {outfit}, hand on heart, sincere warm emotional expression, soft golden lighting, close up portrait, 4K',
      environment: 'warm_light',
      emotion: 'sincere',
      action: 'hand_on_heart',
    },
    curious: {
      prompt: '{triggerWord}, woman with {hair} and {eyes}, wearing {outfit}, tilting head thoughtfully with curious expression, examining something in hands, soft natural lighting, 4K',
      environment: 'natural',
      emotion: 'curious',
      action: 'examining',
    },
    excited: {
      prompt: '{triggerWord}, woman with {hair} and {eyes}, wearing {outfit}, eyes wide with excitement, big joyful smile, hands raised in excitement, bright cheerful lighting, 4K',
      environment: 'bright',
      emotion: 'excited',
      action: 'hands_up',
    },
    thoughtful: {
      prompt: '{triggerWord}, woman with {hair} and {eyes}, wearing {outfit}, hand to chin, looking up thoughtfully, contemplative expression, soft library lighting, 4K',
      environment: 'library',
      emotion: 'thoughtful',
      action: 'thinking',
    },
  },
};


