/**
 * HeyGen Engine Adapter
 * 
 * Generates lip-sync videos using HeyGen's Talking Photo API.
 * Currently BLOCKED (401 Unauthorized) - jobs are pre-queued.
 */

import type { EngineAdapter, VideoJob, EngineSubmitResult, EngineStatusResult } from './types';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY;
const HEYGEN_API_BASE = 'https://api.heygen.com';

export const heygenAdapter: EngineAdapter = {
  name: 'heygen',
  displayName: 'HeyGen',
  
  async submit(job: VideoJob): Promise<EngineSubmitResult> {
    if (!HEYGEN_API_KEY) {
      throw new Error('HEYGEN_API_KEY not configured');
    }
    
    const payload = job.input_payload;
    
    const requestBody = {
      video_inputs: [{
        character: {
          type: 'talking_photo',
          talking_photo_id: payload.talking_photo_id,
        },
        voice: payload.audio_url 
          ? { type: 'audio', audio_url: payload.audio_url }
          : { type: 'text', input_text: payload.script, voice_id: payload.voice_id },
      }],
      dimension: payload.dimension || { width: 1024, height: 1024 },
    };
    
    const response = await fetch(`${HEYGEN_API_BASE}/v2/video/generate`, {
      method: 'POST',
      headers: {
        'X-Api-Key': HEYGEN_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });
    
    if (response.status === 401) {
      throw new Error('BLOCKED:401 - HeyGen API key invalid or expired');
    }
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HeyGen API error: ${response.status} - ${errorText}`);
    }
    
    const data = await response.json();
    
    if (!data.data?.video_id) {
      throw new Error('HeyGen response missing video_id');
    }
    
    return { 
      external_id: data.data.video_id,
      status: 'submitted',
    };
  },
  
  async checkStatus(external_id: string): Promise<EngineStatusResult> {
    if (!HEYGEN_API_KEY) {
      throw new Error('HEYGEN_API_KEY not configured');
    }
    
    const response = await fetch(
      `${HEYGEN_API_BASE}/v1/video_status.get?video_id=${external_id}`,
      {
        headers: { 'X-Api-Key': HEYGEN_API_KEY },
      }
    );
    
    if (response.status === 401) {
      return { status: 'failed', error: 'BLOCKED:401' };
    }
    
    if (!response.ok) {
      const errorText = await response.text();
      return { status: 'failed', error: `API error: ${response.status} - ${errorText}` };
    }
    
    const data = await response.json();
    const heygenStatus = data.data?.status;
    
    // Map HeyGen status to our status
    const statusMap: Record<string, EngineStatusResult['status']> = {
      'pending': 'processing',
      'processing': 'processing',
      'completed': 'completed',
      'failed': 'failed',
    };
    
    return {
      status: statusMap[heygenStatus] || 'processing',
      output_url: data.data?.video_url,
      error: data.data?.error,
    };
  },
  
  async isAvailable(): Promise<boolean> {
    if (!HEYGEN_API_KEY) return false;
    
    try {
      const response = await fetch(`${HEYGEN_API_BASE}/v2/avatars`, {
        headers: { 'X-Api-Key': HEYGEN_API_KEY },
      });
      return response.status === 200;
    } catch {
      return false;
    }
  },
  
  estimateProcessingTime(audioDurationSeconds: number): number {
    // HeyGen typically takes 2-5 minutes per video
    return Math.max(120, audioDurationSeconds * 4);
  },
};

export default heygenAdapter;
