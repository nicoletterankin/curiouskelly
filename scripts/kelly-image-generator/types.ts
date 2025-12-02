/**
 * Kelly Image Generation System - Type Definitions
 * 
 * These types define the complete interface for Kelly's AI image generation
 * and storage system. They're designed to be stable and extensible.
 */

// ═══════════════════════════════════════════════════════════════════════════
// CORE TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Image types that can be generated for Kelly
 */
export type ImageType = 
  // Base poses (universal, topic-agnostic)
  | 'welcome'
  | 'thinking' 
  | 'explaining'
  | 'listening'
  | 'excited'
  | 'celebrating'
  | 'encouraging'
  | 'curious'
  | 'pointing_left'
  | 'pointing_right'
  | 'waving'
  // Lesson-specific types
  | 'hero'
  | 'intro'
  | 'q1' | 'q2' | 'q3'
  | 'hook'
  | 'wisdom'
  | 'reaction_correct'
  | 'reaction_incorrect';

/**
 * Categories for lesson topics - determines props and context
 */
export type LessonCategory = 
  | 'science'
  | 'philosophy'
  | 'creativity'
  | 'nature'
  | 'emotion'
  | 'society'
  | 'health'
  | 'technology'
  | 'history'
  | 'culture';

/**
 * Supported AI image generators
 */
export type GeneratorName = 
  | 'flux-1.1-pro'
  | 'dall-e-3'
  | 'stable-diffusion-xl'
  | 'midjourney';

/**
 * Generation job status
 */
export type JobStatus = 
  | 'pending'
  | 'processing'
  | 'completed'
  | 'failed'
  | 'cancelled';

// ═══════════════════════════════════════════════════════════════════════════
// CHARACTER REFERENCE
// ═══════════════════════════════════════════════════════════════════════════

/**
 * The character reference that defines Kelly's visual identity
 */
export interface CharacterReference {
  id: string;
  version: string;
  description?: string;
  isActive: boolean;
  
  /** URLs to reference images in Supabase Storage */
  referenceImages: string[];
  
  /** The master prompt describing Kelly */
  stylePrompt: string;
  
  /** What to avoid in generation */
  negativePrompt: string;
  
  /** Optional face embedding for consistency checking */
  faceEmbedding?: {
    model: string;
    vector: number[];
  };
  
  createdAt: string;
  updatedAt: string;
}

// ═══════════════════════════════════════════════════════════════════════════
// PROMPT TEMPLATES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * A reusable prompt template with variable substitution
 */
export interface PromptTemplate {
  id: string;
  name: string;
  category: 'base_pose' | 'lesson_specific' | 'reaction';
  description?: string;
  
  /** Template using {{variable}} syntax */
  promptTemplate: string;
  
  /** Variables that must be provided */
  requiredVariables: string[];
  
  /** Default negative prompt for this template */
  defaultNegativePrompt?: string;
  
  /** Default generation parameters */
  defaultParams?: GenerationParams;
  
  version: number;
  isActive: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════
// KELLY IMAGES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * A generated Kelly image stored in the system
 */
export interface KellyImage {
  id: string;
  
  // Identity
  imageType: 'base_pose' | 'lesson_specific' | 'reaction';
  state: ImageType;
  
  // Lesson context (for lesson-specific images)
  lessonDay?: number;
  lessonTopic?: string;
  lessonCategory?: LessonCategory;
  
  // Storage
  storageBucket: string;
  storagePath: string;
  publicUrl: string;
  thumbnailPath?: string;
  thumbnailUrl?: string;
  
  // Generation info
  characterRefId: string;
  templateId?: string;
  fullPrompt: string;
  negativePrompt?: string;
  generator: GeneratorName;
  modelVersion?: string;
  seed?: number;
  generationParams?: GenerationParams;
  
  // Quality
  qualityScore?: number;
  consistencyScore?: number;
  autoApproved: boolean;
  isApproved: boolean;
  approvedBy?: string;
  approvedAt?: string;
  rejectionReason?: string;
  
  // Technical
  width: number;
  height: number;
  fileSizeBytes?: number;
  format: 'png' | 'webp' | 'jpeg';
  
  // Analytics
  viewCount: number;
  engagementScore?: number;
  
  createdAt: string;
  updatedAt: string;
}

// ═══════════════════════════════════════════════════════════════════════════
// GENERATION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Parameters for image generation
 */
export interface GenerationParams {
  /** Number of inference steps (higher = better quality, slower) */
  steps?: number;
  
  /** Guidance scale (how closely to follow prompt) */
  guidance?: number;
  
  /** Output dimensions */
  width?: number;
  height?: number;
  
  /** Seed for reproducibility */
  seed?: number;
  
  /** Provider-specific params */
  [key: string]: unknown;
}

/**
 * Request to generate Kelly images
 */
export interface GenerationRequest {
  /** Type of job */
  jobType: 'lesson_batch' | 'single' | 'regenerate';
  
  /** Target lesson (null for base poses) */
  lessonDay?: number;
  
  /** Which image types to generate */
  imageTypes: ImageType[];
  
  /** Priority (1 = highest) */
  priority?: number;
  
  /** Character reference to use */
  characterRefId: string;
  
  /** Which generator to use */
  generator: GeneratorName;
  
  /** Variables for prompt templates */
  promptVariables: Record<string, string>;
  
  /** Additional options */
  options?: {
    /** Generate multiple variations to choose from */
    variations?: number;
    
    /** Quality preset */
    quality?: 'draft' | 'standard' | 'premium';
    
    /** Specific seed for reproducibility */
    seed?: number;
  };
}

/**
 * A job in the generation queue
 */
export interface GenerationJob {
  id: string;
  
  // Spec
  jobType: 'lesson_batch' | 'single' | 'regenerate';
  lessonDay?: number;
  imageTypes: ImageType[];
  priority: number;
  
  // Params
  characterRefId: string;
  generator: GeneratorName;
  promptVariables: Record<string, string>;
  generationOptions: GenerationRequest['options'];
  
  // Status
  status: JobStatus;
  progress: number;
  currentStep?: string;
  errorMessage?: string;
  errorDetails?: Record<string, unknown>;
  
  // Results
  generatedImageIds: string[];
  approvedImageIds: string[];
  rejectedImageIds: string[];
  
  // Timing
  createdAt: string;
  scheduledFor?: string;
  startedAt?: string;
  completedAt?: string;
  
  // Retry
  attemptCount: number;
  maxAttempts: number;
  nextRetryAt?: string;
  
  // Metadata
  createdBy?: string;
  notes?: string;
}

/**
 * Result from a generation operation
 */
export interface GenerationResult {
  jobId: string;
  status: JobStatus;
  
  /** Generated images (may include multiple variations) */
  images: GeneratedImageResult[];
  
  /** Usage/cost tracking */
  usage: {
    generator: GeneratorName;
    imagesGenerated: number;
    tokensUsed?: number;
    computeSeconds: number;
    estimatedCostUsd: number;
  };
  
  /** Any errors that occurred */
  errors?: Array<{
    imageType: ImageType;
    error: string;
    retryable: boolean;
  }>;
}

/**
 * A single generated image result
 */
export interface GeneratedImageResult {
  imageType: ImageType;
  imageId: string;
  url: string;
  thumbnailUrl?: string;
  
  /** The prompt that was used */
  prompt: string;
  seed: number;
  
  /** Quality metrics */
  qualityScore: number;
  consistencyScore: number;
  
  /** Auto-approval result */
  autoApproved: boolean;
  issues?: string[];
}

// ═══════════════════════════════════════════════════════════════════════════
// QUALITY CONTROL
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Result from quality control check
 */
export interface QualityCheckResult {
  /** Overall quality score (0-1) */
  score: number;
  
  /** Whether the image was auto-approved */
  autoApproved: boolean;
  
  /** Whether it needs human review */
  requiresHumanReview: boolean;
  
  /** Whether it was auto-rejected */
  autoRejected: boolean;
  
  /** Specific issues found */
  issues: QualityIssue[];
  
  /** Individual check results */
  checks: QualityCheck[];
}

export interface QualityCheck {
  name: string;
  passed: boolean;
  score: number;
  reason?: string;
}

export interface QualityIssue {
  type: 'consistency' | 'quality' | 'content' | 'technical';
  severity: 'low' | 'medium' | 'high';
  description: string;
  autoFixable: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════
// CLIENT SDK
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Options for getting a Kelly image
 */
export interface GetImageOptions {
  /** Prefer lesson-specific over base pose */
  preferLessonSpecific?: boolean;
  
  /** Return thumbnail URL instead of full */
  thumbnail?: boolean;
  
  /** Skip cache */
  noCache?: boolean;
}

/**
 * Result from getting a Kelly image
 */
export interface GetImageResult {
  /** The URL to use */
  url: string;
  
  /** Thumbnail URL if available */
  thumbnailUrl?: string;
  
  /** Where the image came from */
  source: 'lesson_specific' | 'base_pose' | 'local_fallback' | 'cache';
  
  /** Image ID for analytics */
  imageId?: string;
  
  /** Whether this is a fallback */
  isFallback: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════
// LESSON CONTEXT
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Lesson data needed for image generation
 */
export interface LessonContext {
  dayNumber: number;
  topic: string;
  category: LessonCategory;
  universalTruth: string;
  keyTerms: string[];
  
  /** Phase-specific content for prompt context */
  phases?: {
    q1?: { question: string; options: string[] };
    q2?: { question: string; options: string[] };
    q3?: { question: string; options: string[] };
    hook?: { content: string };
    wisdom?: { content: string };
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// PROP LIBRARY
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Props that Kelly can hold/interact with by category
 */
export const PROP_LIBRARY: Record<LessonCategory, string[]> = {
  science: [
    "a small magnifying glass, held thoughtfully",
    "a molecular model floating nearby",
    "a miniature telescope",
    "a beaker with colorful liquid",
    "a small globe showing Earth",
  ],
  philosophy: [
    "an antique leather-bound book with visible pages",
    "a small balance scale",
    "a glowing light bulb representing ideas",
    "a compass suggesting direction and choices",
    "a small hourglass",
  ],
  creativity: [
    "an artist's paint palette with vibrant colors",
    "musical notes floating playfully nearby",
    "colorful geometric shapes",
    "a small canvas with abstract art",
    "colored pencils arranged artfully",
  ],
  nature: [
    "a small potted plant with green leaves",
    "a beautiful butterfly resting nearby",
    "a smooth river stone",
    "a small terrarium",
    "a blooming flower",
  ],
  emotion: [
    "a small heart symbol",
    "theatrical masks showing emotions",
    "a mirror reflecting warmth",
    "hands clasped in connection",
    "a small journal",
  ],
  society: [
    "connected puzzle pieces",
    "a small model of a community",
    "hands of different sizes together",
    "a voting ballot",
    "a small globe with people icons",
  ],
  health: [
    "a red apple",
    "a small dumbbell",
    "a heart rate symbol",
    "a peaceful meditation pose icon",
    "a water droplet",
  ],
  technology: [
    "a glowing circuit pattern",
    "a small robot helper",
    "floating digital icons",
    "a holographic display",
    "interconnected nodes",
  ],
  history: [
    "an ancient scroll",
    "a small hourglass",
    "historical photographs floating nearby",
    "an antique key",
    "a world map with historical routes",
  ],
  culture: [
    "diverse cultural symbols",
    "traditional art elements",
    "musical instruments from various cultures",
    "a tapestry of patterns",
    "festival celebration elements",
  ],
};

// ═══════════════════════════════════════════════════════════════════════════
// IMAGE TYPE FALLBACKS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Maps lesson-specific image types to base pose fallbacks
 */
export const IMAGE_TYPE_FALLBACKS: Record<ImageType, ImageType> = {
  // Base poses fallback to themselves
  welcome: 'welcome',
  thinking: 'thinking',
  explaining: 'explaining',
  listening: 'listening',
  excited: 'excited',
  celebrating: 'celebrating',
  encouraging: 'encouraging',
  curious: 'curious',
  pointing_left: 'pointing_left',
  pointing_right: 'pointing_right',
  waving: 'waving',
  
  // Lesson-specific fallback to base poses
  hero: 'welcome',
  intro: 'welcome',
  q1: 'thinking',
  q2: 'thinking',
  q3: 'thinking',
  hook: 'excited',
  wisdom: 'explaining',
  reaction_correct: 'celebrating',
  reaction_incorrect: 'encouraging',
};



