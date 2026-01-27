/**
 * Fal.ai SadTalker Engine Adapter
 * 
 * Good quality lip-sync from still images.
 * Fast processing, works well with portraits.
 */

import type { EngineAdapter, VideoJob, EngineSubmitResult, EngineStatusResult } from './types';

const FAL_KEY = process.env.FAL_KEY;
const FAL_API_BASE = 'https://queue.fal.run';

export const falSadtalkerAdapter: EngineAdapter = {
  name: 'fal_sadtalker',
  displayName: 'Fal.ai SadTalker',
  
  async submit(job: VideoJob): Promise<EngineSubmitResult> {
    if (!FAL_KEY) {
      throw new Error('FAL_KEY not configured');
    }
    
    const payload = job.input_payload;
    
    const requestBody = {
      source_image_url: payload.source_image_url || payload.image_url,
      driven_audio_url: payload.driven_audio_url || payload.audio_url,
      still_mode: true,
      preprocess: 'crop',
      enhancer: 'gfpgan',
    };
    
    const response = await fetch(`${FAL_API_BASE}/fal-ai/sadtalker`, {
      method: 'POST',
      headers: {
        'Authorization': `Key ${FAL_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Fal.ai SadTalker error: ${response.status} - ${errorText}`);
    }
    
    const data = await response.json();
    
    return {
      external_id: data.request_id,
      status: 'submitted',
    };
  },
  
  async checkStatus(external_id: string): Promise<EngineStatusResult> {
    if (!FAL_KEY) {
      throw new Error('FAL_KEY not configured');
    }
    
    const response = await fetch(
      `${FAL_API_BASE}/fal-ai/sadtalker/requests/${external_id}/status`,
      {
        headers: { 'Authorization': `Key ${FAL_KEY}` },
      }
    );
    
    if (!response.ok) {
      return { status: 'failed', error: `Status check failed: ${response.status}` };
    }
    
    const data = await response.json();
    
    if (data.status === 'COMPLETED') {
      // Fetch the result
      const resultResponse = await fetch(
        `${FAL_API_BASE}/fal-ai/sadtalker/requests/${external_id}`,
        {
          headers: { 'Authorization': `Key ${FAL_KEY}` },
        }
      );
      const result = await resultResponse.json();
      
      return {
        status: 'completed',
        output_url: result.video?.url || result.video_url,
      };
    }
    
    if (data.status === 'FAILED') {
      return { status: 'failed', error: data.error || 'Unknown error' };
    }
    
    return { 
      status: 'processing',
      progress: data.progress,
    };
  },
  
  async isAvailable(): Promise<boolean> {
    if (!FAL_KEY) return false;
    return true; // SadTalker is generally available when FAL_KEY is set
  },
  
  estimateProcessingTime(audioDurationSeconds: number): number {
    // SadTalker is fast: ~20-40 seconds per 15s of audio
    return Math.max(20, audioDurationSeconds * 2);
  },
};

export default falSadtalkerAdapter;
