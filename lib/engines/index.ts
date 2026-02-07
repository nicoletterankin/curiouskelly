/**
 * Engine Registry
 * 
 * Central access to all video generation engine adapters.
 */

import type { EngineAdapter, EngineType } from './types';
import { heygenAdapter } from './heygen';
import { falLatentsyncAdapter } from './fal-latentsync';
import { falSadtalkerAdapter } from './fal-sadtalker';
import { syncSoAdapter } from './sync-so';
import { replicateAdapter } from './replicate';
import { musetalkLocalAdapter } from './musetalk-local';

export * from './types';

export const engines: Record<EngineType, EngineAdapter> = {
  heygen: heygenAdapter,
  fal_latentsync: falLatentsyncAdapter,
  fal_sadtalker: falSadtalkerAdapter,
  sync_so: syncSoAdapter,
  replicate: replicateAdapter,
  musetalk_local: musetalkLocalAdapter,
};

export function getEngine(type: EngineType): EngineAdapter {
  const engine = engines[type];
  if (!engine) {
    throw new Error(`Unknown engine type: ${type}`);
  }
  return engine;
}

export async function getEngineStatus(): Promise<Record<EngineType, { available: boolean; displayName: string }>> {
  const status: Record<string, { available: boolean; displayName: string }> = {};
  
  for (const [type, adapter] of Object.entries(engines)) {
    try {
      const available = await adapter.isAvailable();
      status[type] = { available, displayName: adapter.displayName };
    } catch {
      status[type] = { available: false, displayName: adapter.displayName };
    }
  }
  
  return status as Record<EngineType, { available: boolean; displayName: string }>;
}

export const ENGINE_TYPES: EngineType[] = [
  'heygen',
  'sync_so',
  'fal_latentsync',
  'fal_sadtalker',
  'replicate',
  'musetalk_local',
];

export { 
  heygenAdapter,
  falLatentsyncAdapter,
  falSadtalkerAdapter,
  syncSoAdapter,
  replicateAdapter,
  musetalkLocalAdapter,
};

// Re-export fallback chain from types
export { PROVIDER_FALLBACK_CHAIN } from './types';
