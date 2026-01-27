/**
 * Sync.so Engine Adapter
 * 
 * High-quality lip-sync using Sync Labs lipsync-2 model.
 * 95% accuracy, professional quality output.
 */

import type { EngineAdapter, VideoJob, EngineSubmitResult, EngineStatusResult } from './types';

const SYNC_LABS_API_KEY = process.env.SYNC_LABS_API_KEY;
const SYNC_API_BASE = 'https://api.sync.so/v2';

export const syncSoAdapter: EngineAdapter = {
  name: 'sync_so',
  displayName: 'Sync.so (LipSync-2)',
  
  async submit(job: VideoJob): Promise<EngineSubmitResult> {
    if (!SYNC_LABS_API_KEY) {
      throw new Error('SYNC_LABS_API_KEY not configured');
    }
    
    const payload = job.input_payload;
    
    const requestBody = {
      model: payload.model || 'lipsync-2',
      input: [
        { type: 'video', url: payload.video_url || payload.source_image_url },
        { type: 'audio', url: payload.audio_url },
      ],
    };
    
    const response = await fetch(`${SYNC_API_BASE}/generate`, {
      method: 'POST',
      headers: {
        'x-api-key': SYNC_LABS_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Sync.so error: ${response.status} - ${errorText}`);
    }
    
    const data = await response.json();
    
    return {
      external_id: data.id,
      status: 'submitted',
    };
  },
  
  async checkStatus(external_id: string): Promise<EngineStatusResult> {
    if (!SYNC_LABS_API_KEY) {
      throw new Error('SYNC_LABS_API_KEY not configured');
    }
    
    const response = await fetch(`${SYNC_API_BASE}/generate/${external_id}`, {
      headers: { 'x-api-key': SYNC_LABS_API_KEY },
    });
    
    if (!response.ok) {
      return { status: 'failed', error: `Status check failed: ${response.status}` };
    }
    
    const data = await response.json();
    
    if (data.status === 'COMPLETED') {
      return {
        status: 'completed',
        output_url: data.output?.[0]?.url || data.outputUrl,
      };
    }
    
    if (data.status === 'FAILED' || data.status === 'REJECTED') {
      return { 
        status: 'failed', 
        error: data.error || data.message || 'Job failed',
      };
    }
    
    return { 
      status: 'processing',
      progress: data.progress,
    };
  },
  
  async isAvailable(): Promise<boolean> {
    if (!SYNC_LABS_API_KEY) return false;
    
    try {
      // Simple health check
      const response = await fetch(`${SYNC_API_BASE}/health`, {
        headers: { 'x-api-key': SYNC_LABS_API_KEY },
      });
      return response.status !== 401;
    } catch {
      return true; // Assume available
    }
  },
  
  estimateProcessingTime(audioDurationSeconds: number): number {
    // Sync.so is slower but high quality: ~2-5 minutes per video
    return Math.max(120, audioDurationSeconds * 5);
  },
};

export default syncSoAdapter;
