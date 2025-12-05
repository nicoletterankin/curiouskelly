/**
 * Kelly Lip-Sync System
 * 
 * Complete lip-sync solution for Curious Kelly, supporting:
 * - Phoneme-based alignment from ElevenLabs audio
 * - Real-time audio analysis for conversations
 * - Expression integration (eyes, brows, emotions)
 * - Unity WebGL bridge
 * 
 * @module lipsync
 */

// Core phoneme-to-viseme mapping
export {
  ARPABET_PHONEMES,
  VISEME_CATEGORIES,
  PHONEME_TO_BLENDSHAPES,
  COARTICULATION_RULES,
  getBlendshapesForPhoneme,
  getVisemeCategory,
  interpolateBlendshapes,
  applyCoarticulation,
  generateBlendshapeTimeline,
} from './phoneme-viseme-map.js';

// Real-time audio analysis
export {
  RealtimeLipSync,
  AudioElementLipSync,
  StreamingLipSync,
  DEFAULT_CONFIG as REALTIME_CONFIG,
} from './realtime-lipsync.js';

// Main orchestrator
export {
  KellyLipSyncOrchestrator,
  DEFAULT_ORCHESTRATOR_CONFIG,
} from './kelly-lipsync-orchestrator.js';

// Default export is the orchestrator
import { KellyLipSyncOrchestrator } from './kelly-lipsync-orchestrator.js';
export default KellyLipSyncOrchestrator;

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/**
 * Quick setup for lesson playback with alignment
 * @param {HTMLAudioElement} audioElement - Audio element
 * @param {Object} alignment - Pre-computed alignment data
 * @param {Object} options - Configuration options
 * @returns {KellyLipSyncOrchestrator} Configured orchestrator
 */
export function setupLessonLipSync(audioElement, alignment, options = {}) {
  const orchestrator = KellyLipSyncOrchestrator.forLessons(options);
  orchestrator.playFromAlignment(alignment, audioElement);
  return orchestrator;
}

/**
 * Quick setup for real-time conversation lip-sync
 * @param {HTMLAudioElement} audioElement - Audio element for Kelly's voice
 * @param {Object} options - Configuration options
 * @returns {KellyLipSyncOrchestrator} Configured orchestrator
 */
export function setupConversationLipSync(audioElement, options = {}) {
  const orchestrator = KellyLipSyncOrchestrator.forConversation(options);
  orchestrator.startRealtimeFromAudio(audioElement);
  return orchestrator;
}

/**
 * Quick setup for ElevenLabs streaming lip-sync
 * @param {Object} options - Configuration options
 * @returns {KellyLipSyncOrchestrator} Configured orchestrator
 */
export function setupStreamingLipSync(options = {}) {
  const orchestrator = KellyLipSyncOrchestrator.forStreaming(options);
  orchestrator.startStreamingLipSync();
  return orchestrator;
}

/**
 * Generate alignment from audio URL and transcript
 * @param {string} audioUrl - URL to audio file
 * @param {string} transcript - Text transcript
 * @returns {Promise<Object>} Alignment data
 */
export async function generateAlignment(audioUrl, transcript) {
  const response = await fetch('/api/align', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ audio_url: audioUrl, transcript }),
  });
  
  if (!response.ok) {
    throw new Error(`Alignment failed: ${response.status}`);
  }
  
  return response.json();
}

// =============================================================================
// INTEGRATION WITH KELLY CONVERSATION
// =============================================================================

/**
 * Integrate lip-sync with KellyConversation system
 * @param {Object} kellyConversation - KellyConversation instance
 * @returns {KellyLipSyncOrchestrator} Connected orchestrator
 */
export function integrateWithConversation(kellyConversation) {
  const orchestrator = KellyLipSyncOrchestrator.forStreaming({
    enableExpressions: true,
    lipSyncWeight: 0.85,
  });
  
  orchestrator.startStreamingLipSync();
  
  // Connect expression callback
  orchestrator.setExpressionCallback(() => {
    // Get current expression from conversation state
    if (kellyConversation.isSpeaking) {
      return { mouthSmileLeft: 20, mouthSmileRight: 20 };
    } else if (kellyConversation.isListening) {
      return { eyebrowRaise: 15, mouthSmileLeft: 25, mouthSmileRight: 25 };
    }
    return {};
  });
  
  // Hook into conversation audio events
  const originalPlayNextAudio = kellyConversation.playNextAudio?.bind(kellyConversation);
  if (originalPlayNextAudio) {
    kellyConversation.playNextAudio = async function() {
      const base64Audio = this.audioQueue.shift();
      if (base64Audio) {
        // Decode and send to lip-sync
        const binaryString = atob(base64Audio);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        orchestrator.addStreamingAudioChunk(bytes.buffer);
      }
      
      // Call original
      if (originalPlayNextAudio) {
        return originalPlayNextAudio();
      }
    };
  }
  
  return orchestrator;
}

// =============================================================================
// VERSION INFO
// =============================================================================

export const VERSION = '1.0.0';
export const BUILD_DATE = '2025-12-04';

console.log(`[KellyLipSync] v${VERSION} loaded`);


