/**
 * Video Engine Types
 * 
 * Unified interface for all lip-sync video generation engines.
 */

export type EngineType = 
  | 'heygen' 
  | 'fal_latentsync' 
  | 'fal_sadtalker' 
  | 'sync_so' 
  | 'musetalk_local';

export type JobStatus = 
  | 'queued' 
  | 'submitted' 
  | 'processing' 
  | 'completed' 
  | 'failed' 
  | 'blocked';

export type AgeCategory = 
  | 'toddler' 
  | 'child' 
  | 'teen' 
  | 'young_adult' 
  | 'adult' 
  | 'elder';

export type Phase = 'hook' | 'story' | 'wonder' | 'action' | 'wisdom';

export interface VideoJob {
  id: string;
  day_of_year: number;
  phase: Phase;
  age_category: AgeCategory;
  language: string;
  engine: EngineType;
  status: JobStatus;
  external_id?: string;
  input_payload: EngineInputPayload;
  output_url?: string;
  quality_score?: number;
  quality_notes?: string;
  is_approved?: boolean;
  error_message?: string;
  priority: number;
  created_at: string;
  updated_at: string;
  submitted_at?: string;
  completed_at?: string;
}

export interface EngineInputPayload {
  // Common fields
  audio_url?: string;
  source_image_url?: string;
  
  // HeyGen specific
  talking_photo_id?: string;
  script?: string;
  voice_id?: string;
  dimension?: { width: number; height: number };
  
  // Fal.ai specific
  driven_audio_url?: string;
  image_url?: string;
  
  // Sync.so specific
  video_url?: string;
  model?: string;
  
  // MuseTalk specific
  local_audio_path?: string;
  local_image_path?: string;
  local_output_path?: string;
}

export interface EngineSubmitResult {
  external_id: string;
  status?: string;
}

export interface EngineStatusResult {
  status: 'processing' | 'completed' | 'failed';
  output_url?: string;
  error?: string;
  progress?: number;
}

export interface EngineAdapter {
  name: EngineType;
  displayName: string;
  
  /**
   * Submit a job to the engine
   */
  submit(job: VideoJob): Promise<EngineSubmitResult>;
  
  /**
   * Check status of a submitted job
   */
  checkStatus(external_id: string): Promise<EngineStatusResult>;
  
  /**
   * Check if engine is currently available (API accessible)
   */
  isAvailable(): Promise<boolean>;
  
  /**
   * Estimate processing time in seconds
   */
  estimateProcessingTime(audioDurationSeconds: number): number;
}

export interface QueueJobRequest {
  day_of_year: number;
  phase: Phase;
  age_category: AgeCategory;
  language?: string;
  engine: EngineType;
  input_payload: EngineInputPayload;
  priority?: number;
}

export interface SubmitJobsRequest {
  job_id?: string;
  engine?: EngineType;
  status?: JobStatus;
  limit?: number;
  dry_run?: boolean;
}

export interface RateJobRequest {
  job_id: string;
  quality_score: number;
  quality_notes?: string;
  is_approved?: boolean;
}

export interface JobCompareResponse {
  day: number;
  phase: Phase;
  age: AgeCategory;
  engines: Record<EngineType, VideoJob | null>;
}

export interface JobStatusResponse {
  jobs: VideoJob[];
  total: number;
  summary: {
    queued: number;
    submitted: number;
    processing: number;
    completed: number;
    failed: number;
    blocked: number;
  };
}

export interface VideoUrlResponse {
  url: string | null;
  engine: EngineType | null;
  quality_score: number | null;
  is_approved: boolean;
  fallback: boolean;
  available_engines: EngineType[];
}
