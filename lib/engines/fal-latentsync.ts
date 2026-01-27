/**
 * Fal.ai LatentSync Engine Adapter
 * 
 * High-quality lip-sync using LatentSync model.
 * Good for realistic face movements.
 */

import type { EngineAdapter, VideoJob, EngineSubmitResult, EngineStatusResult } from './types';

const FAL_KEY = process.env.FAL_KEY;
const FAL_API_BASE = 'https://queue.fal.run';

export const falLatentsyncAdapter: EngineAdapter = {
  name: 'fal_latentsync',
  displayName: 'Fal.ai LatentSync',
  
  async submit(job: VideoJob): Promise<EngineSubmitResult> {
    if (!FAL_KEY) {
      throw new Error('FAL_KEY not configured');
    }
    
    const payload = job.input_payload;
    
    const requestBody = {
      video_url: payload.video_url || payload.source_image_url,
      audio_url: payload.audio_url || payload.driven_audio_url,
    };
    
    const response = await fetch(`${FAL_API_BASE}/fal-ai/latentsync`, {
      method: 'POST',
      headers: {
        'Authorization': `Key ${FAL_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Fal.ai LatentSync error: ${response.status} - ${errorText}`);
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
      `${FAL_API_BASE}/fal-ai/latentsync/requests/${external_id}/status`,
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
        `${FAL_API_BASE}/fal-ai/latentsync/requests/${external_id}`,
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
    
    try {
      // Simple auth check
      const response = await fetch(`${FAL_API_BASE}/fal-ai/latentsync`, {
        method: 'OPTIONS',
        headers: { 'Authorization': `Key ${FAL_KEY}` },
      });
      return response.status !== 401;
    } catch {
      return true; // Assume available if network works
    }
  },
  
  estimateProcessingTime(audioDurationSeconds: number): number {
    // LatentSync is relatively fast: ~30-60 seconds per 15s of audio
    return Math.max(30, audioDurationSeconds * 3);
  },
};

export default falLatentsyncAdapter;
