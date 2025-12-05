import type { VercelRequest, VercelResponse } from '@vercel/node';
import { spawn } from 'child_process';
import { writeFile, unlink, mkdir } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';
import { randomUUID } from 'crypto';

/**
 * Kelly Lip-Sync Alignment API
 * 
 * Generates word-level and phoneme-level timing from audio + transcript.
 * Used for precise lip-sync in lesson playback.
 * 
 * POST /api/align
 * Body: FormData with 'audio' (file) and 'transcript' (text)
 *   OR: JSON with 'audio_url' and 'transcript'
 * 
 * Returns: {
 *   words: [{ word, start, end, confidence }],
 *   phones: [{ phone, start, end, word, viseme }],
 *   duration: number,
 *   method: 'mfa' | 'gentle' | 'estimation',
 *   confidence: number
 * }
 */

// Phoneme to viseme mapping (matches phoneme-viseme-map.js)
const PHONEME_TO_VISEME: Record<string, string> = {
  // Vowels
  'AA': 'A', 'AE': 'A', 'AH': 'A',
  'AO': 'O', 'AW': 'O', 'OW': 'O', 'OY': 'O',
  'EH': 'E', 'EY': 'E',
  'IH': 'I', 'IY': 'I',
  'UH': 'U', 'UW': 'U', 'ER': 'R',
  'AY': 'A',
  
  // Consonants
  'P': 'M', 'B': 'M', 'M': 'M',
  'F': 'F', 'V': 'F',
  'TH': 'C', 'DH': 'C',
  'T': 'C', 'D': 'C', 'N': 'C', 'S': 'C', 'Z': 'C',
  'SH': 'SH', 'ZH': 'SH', 'CH': 'SH', 'JH': 'SH',
  'K': 'C', 'G': 'C', 'NG': 'C', 'HH': 'A',
  'L': 'L', 'R': 'R',
  'W': 'U', 'Y': 'I',
  
  // Silence
  'SIL': 'REST', 'SP': 'REST', 'spn': 'REST',
};

// Letter to phoneme for estimation
const LETTER_TO_PHONEME: Record<string, string> = {
  'a': 'AE', 'e': 'EH', 'i': 'IH', 'o': 'AA', 'u': 'AH',
  'b': 'B', 'c': 'K', 'd': 'D', 'f': 'F', 'g': 'G',
  'h': 'HH', 'j': 'JH', 'k': 'K', 'l': 'L', 'm': 'M',
  'n': 'N', 'p': 'P', 'q': 'K', 'r': 'R', 's': 'S',
  't': 'T', 'v': 'V', 'w': 'W', 'x': 'K', 'y': 'Y', 'z': 'Z',
};

interface WordAlignment {
  word: string;
  start: number;
  end: number;
  confidence: number;
}

interface PhoneAlignment {
  phone: string;
  start: number;
  end: number;
  word: string;
  viseme: string;
}

interface AlignmentResult {
  words: WordAlignment[];
  phones: PhoneAlignment[];
  duration: number;
  method: string;
  confidence: number;
  transcript?: string;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    let transcript: string = '';
    let audioBuffer: Buffer | null = null;
    let audioUrl: string | null = null;

    // Parse request body
    const contentType = req.headers['content-type'] || '';
    
    if (contentType.includes('multipart/form-data')) {
      // Handle FormData (file upload)
      // Note: In production, use a proper multipart parser
      return res.status(400).json({ 
        error: 'File upload not supported in this version. Use audio_url instead.',
        suggestion: 'Upload audio to Supabase storage first, then pass the URL.'
      });
    } else {
      // Handle JSON body
      const body = req.body;
      transcript = body.transcript || body.text || '';
      audioUrl = body.audio_url || body.audioUrl || null;
      
      // If audio data is provided as base64
      if (body.audio_base64) {
        audioBuffer = Buffer.from(body.audio_base64, 'base64');
      }
    }

    // Validate inputs
    if (!transcript) {
      return res.status(400).json({ error: 'transcript is required' });
    }

    console.log('[Align API] Processing:', { 
      transcriptLength: transcript.length,
      hasAudioUrl: !!audioUrl,
      hasAudioBuffer: !!audioBuffer
    });

    let alignment: AlignmentResult;

    // If we have audio, try to run alignment
    if (audioUrl || audioBuffer) {
      try {
        // Try external alignment service (Gentle or similar)
        alignment = await tryExternalAlignment(transcript, audioUrl, audioBuffer);
      } catch (error) {
        console.warn('[Align API] External alignment failed, using estimation:', error);
        alignment = estimateAlignment(transcript);
      }
    } else {
      // No audio provided, use estimation
      console.log('[Align API] No audio provided, using estimation');
      alignment = estimateAlignment(transcript);
    }

    // Add transcript to response
    alignment.transcript = transcript;

    return res.status(200).json(alignment);

  } catch (error) {
    console.error('[Align API] Error:', error);
    return res.status(500).json({ 
      error: error instanceof Error ? error.message : 'Internal server error'
    });
  }
}

/**
 * Try external alignment service (Gentle)
 */
async function tryExternalAlignment(
  transcript: string, 
  audioUrl: string | null, 
  audioBuffer: Buffer | null
): Promise<AlignmentResult> {
  // Try Gentle aligner if available
  const gentleUrl = process.env.GENTLE_API_URL || 'http://localhost:8765';
  
  try {
    // Fetch audio if URL provided
    if (audioUrl && !audioBuffer) {
      const audioResponse = await fetch(audioUrl);
      if (!audioResponse.ok) {
        throw new Error(`Failed to fetch audio: ${audioResponse.status}`);
      }
      audioBuffer = Buffer.from(await audioResponse.arrayBuffer());
    }

    if (!audioBuffer) {
      throw new Error('No audio data available');
    }

    // Call Gentle API
    const formData = new FormData();
    formData.append('audio', new Blob([audioBuffer]), 'audio.wav');
    formData.append('transcript', transcript);

    const response = await fetch(`${gentleUrl}/transcriptions?async=false`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Gentle API error: ${response.status}`);
    }

    const gentleResult = await response.json();
    return convertGentleResult(gentleResult);

  } catch (error) {
    // Gentle not available, fall back to estimation
    throw error;
  }
}

/**
 * Convert Gentle aligner result to standard format
 */
function convertGentleResult(gentleResult: any): AlignmentResult {
  const words: WordAlignment[] = [];
  const phones: PhoneAlignment[] = [];

  for (const wordData of gentleResult.words || []) {
    if (wordData.case === 'success') {
      words.push({
        word: wordData.word,
        start: wordData.start,
        end: wordData.end,
        confidence: 0.9,
      });

      for (const phoneData of wordData.phones || []) {
        const phone = phoneData.phone.split('_')[0].toUpperCase();
        const normalized = normalizePhoneme(phone);
        
        phones.push({
          phone: normalized,
          start: phoneData.start,
          end: phoneData.start + phoneData.duration,
          word: wordData.word,
          viseme: PHONEME_TO_VISEME[normalized] || 'REST',
        });
      }
    }
  }

  return {
    words,
    phones,
    duration: gentleResult.duration || (words.length > 0 ? words[words.length - 1].end : 0),
    method: 'gentle',
    confidence: 0.9,
  };
}

/**
 * Estimate alignment from transcript (no audio)
 */
function estimateAlignment(transcript: string, duration?: number): AlignmentResult {
  const words = transcript.trim().split(/\s+/).filter(w => w.length > 0);
  
  if (words.length === 0) {
    return {
      words: [],
      phones: [],
      duration: 0,
      method: 'estimation',
      confidence: 0,
    };
  }

  // Estimate duration: ~0.4 seconds per word
  const estimatedDuration = duration || words.length * 0.4;
  const wordDuration = estimatedDuration / words.length;
  
  const wordAlignments: WordAlignment[] = [];
  const phoneAlignments: PhoneAlignment[] = [];
  let currentTime = 0;

  for (const word of words) {
    const wordEnd = currentTime + wordDuration;
    
    wordAlignments.push({
      word,
      start: Math.round(currentTime * 1000) / 1000,
      end: Math.round(wordEnd * 1000) / 1000,
      confidence: 0.5,
    });

    // Estimate phonemes from letters
    const phonemes = estimatePhonemesFromWord(word);
    const phoneDuration = wordDuration / phonemes.length;
    
    for (let i = 0; i < phonemes.length; i++) {
      const phone = phonemes[i];
      phoneAlignments.push({
        phone,
        start: Math.round((currentTime + (i * phoneDuration)) * 1000) / 1000,
        end: Math.round((currentTime + ((i + 1) * phoneDuration)) * 1000) / 1000,
        word,
        viseme: PHONEME_TO_VISEME[phone] || 'REST',
      });
    }

    currentTime = wordEnd + 0.05; // Small gap between words
  }

  return {
    words: wordAlignments,
    phones: phoneAlignments,
    duration: estimatedDuration,
    method: 'estimation',
    confidence: 0.5,
  };
}

/**
 * Estimate phonemes from word spelling
 */
function estimatePhonemesFromWord(word: string): string[] {
  const phonemes: string[] = [];
  const lowerWord = word.toLowerCase().replace(/[^a-z]/g, '');
  
  for (let i = 0; i < lowerWord.length; i++) {
    const char = lowerWord[i];
    
    // Handle digraphs
    if (i < lowerWord.length - 1) {
      const digraph = lowerWord.substring(i, i + 2);
      if (digraph === 'th') {
        phonemes.push('TH');
        i++;
        continue;
      } else if (digraph === 'sh') {
        phonemes.push('SH');
        i++;
        continue;
      } else if (digraph === 'ch') {
        phonemes.push('CH');
        i++;
        continue;
      } else if (digraph === 'ng') {
        phonemes.push('NG');
        i++;
        continue;
      }
    }
    
    if (LETTER_TO_PHONEME[char]) {
      phonemes.push(LETTER_TO_PHONEME[char]);
    }
  }

  return phonemes.length > 0 ? phonemes : ['SIL'];
}

/**
 * Normalize phoneme (remove stress markers)
 */
function normalizePhoneme(phone: string): string {
  return phone.replace(/[0-9]/g, '').toUpperCase();
}


