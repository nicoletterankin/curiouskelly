/**
 * MuseTalk Local Engine Adapter
 * 
 * Runs locally on GPU machine (RTX 5090).
 * No API calls - processes via local filesystem.
 * Jobs are queued and picked up by local worker.
 */

import type { EngineAdapter, VideoJob, EngineSubmitResult, EngineStatusResult } from './types';

// MuseTalk output directory (shared between API and local machine)
const MUSETALK_OUTPUT_BASE = process.env.MUSETALK_OUTPUT_BASE || 
  'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/videos/lipsync';

export const musetalkLocalAdapter: EngineAdapter = {
  name: 'musetalk_local',
  displayName: 'MuseTalk (Local GPU)',
  
  async submit(job: VideoJob): Promise<EngineSubmitResult> {
    // MuseTalk jobs are "submitted" by being queued in the database
    // The local worker picks them up and processes them
    // We generate a unique job reference
    
    const jobRef = `musetalk_${job.day_of_year}_${job.phase}_${Date.now()}`;
    
    // The job payload should contain:
    // - local_audio_path or audio_url
    // - local_image_path or source_image_url
    // - local_output_path (where to save result)
    
    return {
      external_id: jobRef,
      status: 'queued',
    };
  },
  
  async checkStatus(external_id: string): Promise<EngineStatusResult> {
    // For MuseTalk, we check if the output file exists in blob storage
    // The external_id contains info about day/phase
    
    // Parse job reference: musetalk_DAY_PHASE_TIMESTAMP
    const parts = external_id.split('_');
    if (parts.length < 3) {
      return { status: 'failed', error: 'Invalid job reference' };
    }
    
    const day = parts[1];
    const phase = parts[2];
    
    // Check if video exists at expected location
    const expectedUrl = `${MUSETALK_OUTPUT_BASE}/2026/en/day-${day.padStart(3, '0')}/${phase}-musetalk.mp4`;
    
    try {
      const response = await fetch(expectedUrl, { method: 'HEAD' });
      
      if (response.ok) {
        return {
          status: 'completed',
          output_url: expectedUrl,
        };
      }
      
      // Still processing (file not yet uploaded)
      return { status: 'processing' };
      
    } catch {
      return { status: 'processing' };
    }
  },
  
  async isAvailable(): Promise<boolean> {
    // MuseTalk is always "available" as jobs queue for local processing
    // Actual availability depends on local GPU machine being online
    return true;
  },
  
  estimateProcessingTime(audioDurationSeconds: number): number {
    // MuseTalk on RTX 5090 is fast: ~15-30 seconds per 15s of audio
    // But includes manual upload step
    return Math.max(60, audioDurationSeconds * 2);
  },
};

export default musetalkLocalAdapter;
