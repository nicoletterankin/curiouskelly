/**
 * Providers Health Check API
 * GET /api/providers/health
 * 
 * Returns availability status for all video generation providers.
 * Used for monitoring and fallback decision-making.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getEngineStatus, ENGINE_TYPES, PROVIDER_FALLBACK_CHAIN } from '../../lib/engines';
import { cors } from '../../lib/cors';

export interface ProviderHealth {
  name: string;
  displayName: string;
  available: boolean;
  inFallbackChain: boolean;
  fallbackOrder?: number;
  lastChecked: string;
}

export interface ProvidersHealthResponse {
  timestamp: string;
  providers: ProviderHealth[];
  summary: {
    total: number;
    available: number;
    unavailable: number;
    fallbackChainHealthy: boolean;
  };
  fallbackChain: string[];
  recommendations: string[];
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Public CORS (monitoring tools need access)
  if (!cors(req, res, { allowAllOrigins: true })) return;
  
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  try {
    const now = new Date().toISOString();
    
    // Check all engine statuses
    const engineStatus = await getEngineStatus();
    
    const providers: ProviderHealth[] = [];
    let availableCount = 0;
    
    for (const engineType of ENGINE_TYPES) {
      const status = engineStatus[engineType];
      const isAvailable = status?.available ?? false;
      const fallbackIndex = PROVIDER_FALLBACK_CHAIN.indexOf(engineType);
      
      if (isAvailable) availableCount++;
      
      providers.push({
        name: engineType,
        displayName: status?.displayName || engineType,
        available: isAvailable,
        inFallbackChain: fallbackIndex >= 0,
        fallbackOrder: fallbackIndex >= 0 ? fallbackIndex + 1 : undefined,
        lastChecked: now,
      });
    }
    
    // Sort by fallback order (if applicable), then by name
    providers.sort((a, b) => {
      if (a.fallbackOrder && b.fallbackOrder) {
        return a.fallbackOrder - b.fallbackOrder;
      }
      if (a.fallbackOrder) return -1;
      if (b.fallbackOrder) return 1;
      return a.name.localeCompare(b.name);
    });
    
    // Check if fallback chain has at least one available provider
    const fallbackChainHealthy = PROVIDER_FALLBACK_CHAIN.some(
      engine => engineStatus[engine]?.available
    );
    
    // Generate recommendations
    const recommendations: string[] = [];
    
    if (!fallbackChainHealthy) {
      recommendations.push('CRITICAL: No providers available in fallback chain. Pipeline cannot generate videos.');
    }
    
    if (availableCount === 0) {
      recommendations.push('CRITICAL: All providers are unavailable. Check API keys and service status.');
    } else if (availableCount === 1) {
      recommendations.push('WARNING: Only one provider available. Consider adding backup provider.');
    }
    
    // Check specific providers
    if (!engineStatus['heygen']?.available) {
      recommendations.push('HeyGen unavailable - jobs will fall back to Sync.so or FAL.');
    }
    
    if (!engineStatus['sync_so']?.available && !engineStatus['heygen']?.available) {
      recommendations.push('WARNING: Both primary providers (HeyGen, Sync.so) unavailable.');
    }
    
    const response: ProvidersHealthResponse = {
      timestamp: now,
      providers,
      summary: {
        total: providers.length,
        available: availableCount,
        unavailable: providers.length - availableCount,
        fallbackChainHealthy,
      },
      fallbackChain: PROVIDER_FALLBACK_CHAIN,
      recommendations,
    };
    
    // Cache for 60 seconds
    res.setHeader('Cache-Control', 's-maxage=60, stale-while-revalidate=120');
    
    return res.status(200).json(response);
    
  } catch (error) {
    console.error('Providers health error:', error);
    return res.status(500).json({
      error: 'Failed to check provider health',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}
