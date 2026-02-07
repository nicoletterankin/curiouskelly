/**
 * Replicate Engine Adapter
 * 
 * Overflow video generation using Replicate's Wav2Lip model.
 * Good for high-volume processing when other providers are at capacity.
 */

import type { EngineAdapter, VideoJob, EngineSubmitResult, EngineStatusResult } from './types';

const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;
const REPLICATE_API_BASE = 'https://api.replicate.com/v1';

// Wav2Lip model for lip-sync
const WAV2LIP_MODEL = 'cjwbw/video-retalking:af3c3f96b7db2b27bf8ad9a3bed97f925c2cffe9e42ea1dd45e66ac5e0e81b3d';

export const replicateAdapter: EngineAdapter = {
  name: 'replicate',
  displayName: 'Replicate (Wav2Lip)',
  
  async submit(job: VideoJob): Promise<EngineSubmitResult> {
    if (!REPLICATE_API_TOKEN) {
      throw new Error('REPLICATE_API_TOKEN not configured');
    }
    
    const payload = job.input_payload;
    
    const requestBody = {
      version: WAV2LIP_MODEL.split(':')[1],
      input: {
        face: payload.video_url || payload.source_image_url,
        audio: payload.audio_url,
      },
    };
    
    const response = await fetch(`${REPLICATE_API_BASE}/predictions`, {
      method: 'POST',
      headers: {
        'Authorization': `Token ${REPLICATE_API_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });
    
    if (response.status === 401) {
      throw new Error('BLOCKED:401 - Replicate API token invalid');
    }
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Replicate error: ${response.status} - ${errorText}`);
    }
    
    const data = await response.json();
    
    return {
      external_id: data.id,
      status: 'submitted',
    };
  },
  
  async checkStatus(external_id: string): Promise<EngineStatusResult> {
    if (!REPLICATE_API_TOKEN) {
      throw new Error('REPLICATE_API_TOKEN not configured');
    }
    
    const response = await fetch(`${REPLICATE_API_BASE}/predictions/${external_id}`, {
      headers: { 'Authorization': `Token ${REPLICATE_API_TOKEN}` },
    });
    
    if (!response.ok) {
      return { status: 'failed', error: `Status check failed: ${response.status}` };
    }
    
    const data = await response.json();
    
    if (data.status === 'succeeded') {
      return {
        status: 'completed',
        output_url: Array.isArray(data.output) ? data.output[0] : data.output,
      };
    }
    
    if (data.status === 'failed' || data.status === 'canceled') {
      return { 
        status: 'failed', 
        error: data.error || 'Job failed',
      };
    }
    
    return { 
      status: 'processing',
      progress: data.progress,
    };
  },
  
  async isAvailable(): Promise<boolean> {
    if (!REPLICATE_API_TOKEN) return false;
    
    try {
      const response = await fetch(`${REPLICATE_API_BASE}/predictions`, {
        method: 'GET',
        headers: { 'Authorization': `Token ${REPLICATE_API_TOKEN}` },
      });
      return response.status === 200;
    } catch {
      return true; // Assume available
    }
  },
  
  estimateProcessingTime(audioDurationSeconds: number): number {
    // Replicate is moderate speed: ~1-3 minutes per video
    return Math.max(60, audioDurationSeconds * 4);
  },
};

export default replicateAdapter;
