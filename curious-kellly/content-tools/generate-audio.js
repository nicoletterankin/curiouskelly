#!/usr/bin/env node

/**
 * Audio Generator
 * Generates audio for lesson content using ElevenLabs or OpenAI TTS
 */

const fs = require('fs');
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '../backend/.env') });

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

// Voice IDs for each Kelly age (ElevenLabs)
// Using Kelly25 voice (wAdymQH5YucAkXwmrdL0) for all ages - voice characteristics adjusted via settings
const KELLY_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';

// OpenAI TTS voices (fallback)
const OPENAI_VOICES = {
  3: 'nova',    // Warm and friendly
  9: 'alloy',   // Balanced
  15: 'echo',   // Natural
  27: 'shimmer', // Clear
  48: 'onyx',   // Authoritative
  82: 'fable',  // Expressive
};

async function generateAudioWithElevenLabs(text, voiceId, outputPath, language = 'en', kellyAge = 27) {
  console.log(`  🎙️  Generating with ElevenLabs (voice: ${voiceId}, language: ${language})...`);
  
  try {
    // Use multilingual model for ES/FR, monolingual for EN
    const modelId = language === 'en' ? 'eleven_monolingual_v1' : 'eleven_multilingual_v2';
    
    // Voice settings adjusted by age
    const voiceSettings = {
      stability: 0.5,
      similarity_boost: 0.75,
      style: 0.0,
      use_speaker_boost: true
    };
    
    // Adjust settings based on Kelly age
    if (kellyAge === 3) {
      voiceSettings.stability = 0.4; // More variation for toddler
      voiceSettings.similarity_boost = 0.7;
    } else if (kellyAge >= 48) {
      voiceSettings.stability = 0.6; // More stable for mature voices
      voiceSettings.similarity_boost = 0.8;
    }
    
    const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`, {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'xi-api-key': ELEVENLABS_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: text,
        model_id: modelId,
        voice_settings: voiceSettings,
      }),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`ElevenLabs API error: ${response.statusText} - ${errorText}`);
    }
    
    const audioBuffer = await response.arrayBuffer();
    fs.writeFileSync(outputPath, Buffer.from(audioBuffer));
    
    console.log(`  ✅ Audio saved: ${outputPath}`);
    return true;
  } catch (error) {
    console.error(`  ❌ ElevenLabs error: ${error.message}`);
    return false;
  }
}

async function generateAudioWithOpenAI(text, voice, outputPath) {
  console.log(`  🎙️  Generating with OpenAI TTS (voice: ${voice})...`);
  
  try {
    const OpenAI = require('openai');
    const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
    
    const mp3 = await openai.audio.speech.create({
      model: 'tts-1',
      voice: voice,
      input: text,
    });
    
    const buffer = Buffer.from(await mp3.arrayBuffer());
    fs.writeFileSync(outputPath, buffer);
    
    console.log(`  ✅ Audio saved: ${outputPath}`);
    return true;
  } catch (error) {
    console.error(`  ❌ OpenAI TTS error: ${error.message}`);
    return false;
  }
}

async function generateLessonAudio(lessonPath, ageGroup = null, useOpenAI = false) {
  console.log(`\n🎙️  Generating audio for: ${lessonPath}\n`);
  
  // Load lesson
  if (!fs.existsSync(lessonPath)) {
    console.error(`❌ File not found: ${lessonPath}`);
    return false;
  }
  
  const lessonData = JSON.parse(fs.readFileSync(lessonPath, 'utf8'));
  const lessonId = lessonData.id;
  
  // Create audio output directory
  const audioDir = path.join(path.dirname(lessonPath), '..', 'audio', lessonId);
  if (!fs.existsSync(audioDir)) {
    fs.mkdirSync(audioDir, { recursive: true });
  }
  
  // Determine which age groups to process
  const ageGroups = ageGroup ? [ageGroup] : Object.keys(lessonData.ageVariants);
  
  let successCount = 0;
  let totalCount = 0;
  
  // Get all available languages from first variant
  const firstVariant = lessonData.ageVariants[Object.keys(lessonData.ageVariants)[0]];
  const availableLanguages = firstVariant.language ? Object.keys(firstVariant.language) : ['en'];
  
  console.log(`📚 Languages to generate: ${availableLanguages.join(', ')}\n`);
  
  for (const ag of ageGroups) {
    const variant = lessonData.ageVariants[ag];
    if (!variant) {
      console.warn(`⚠️  Age group ${ag} not found in lesson`);
      continue;
    }
    
    const kellyAge = variant.kellyAge;
    
    console.log(`\n📝 Age group: ${ag} (Kelly age: ${kellyAge})`);
    
    // Generate audio for each language
    for (const lang of availableLanguages) {
      const content = variant.language[lang];
      if (!content) {
        console.warn(`  ⚠️  Language ${lang} not found for age ${ag}`);
        continue;
      }
      
      console.log(`  🌍 Language: ${lang.toUpperCase()}`);
      
      // Generate audio for each section
      const sections = {
        welcome: content.welcome,
        mainContent: content.mainContent,
        wisdomMoment: content.wisdomMoment,
      };
      
      for (const [section, text] of Object.entries(sections)) {
        if (!text) continue;
        
        totalCount++;
        const outputPath = path.join(audioDir, `${ag}-${section}-${lang}.mp3`);
        
        // Skip if file already exists
        if (fs.existsSync(outputPath)) {
          console.log(`  ⏭️  Skipping ${section} (already exists)`);
          successCount++;
          continue;
        }
        
        let success = false;
        
        if (useOpenAI) {
          const voice = OPENAI_VOICES[kellyAge];
          success = await generateAudioWithOpenAI(text, voice, outputPath);
        } else if (ELEVENLABS_API_KEY) {
          success = await generateAudioWithElevenLabs(text, KELLY_VOICE_ID, outputPath, lang, kellyAge);
          
          // Fallback to OpenAI if ElevenLabs fails
          if (!success && OPENAI_API_KEY) {
            console.log(`  🔄 Falling back to OpenAI TTS...`);
            const voice = OPENAI_VOICES[kellyAge];
            success = await generateAudioWithOpenAI(text, voice, outputPath);
          }
        } else if (OPENAI_API_KEY) {
          const voice = OPENAI_VOICES[kellyAge];
          success = await generateAudioWithOpenAI(text, voice, outputPath);
        } else {
          console.error('❌ No API keys configured (ELEVENLABS_API_KEY or OPENAI_API_KEY)');
          return false;
        }
        
        if (success) successCount++;
        
        // Small delay to avoid rate limiting
        await new Promise(resolve => setTimeout(resolve, 200));
      }
    }
  }
  
  console.log(`\n✅ Audio generation complete: ${successCount}/${totalCount} files generated\n`);
  return successCount === totalCount;
}

// CLI
if (require.main === module) {
  const lessonPath = process.argv[2];
  const ageGroupFlag = process.argv.indexOf('--age-group');
  const ageGroup = ageGroupFlag > -1 ? process.argv[ageGroupFlag + 1] : null;
  const useOpenAI = process.argv.includes('--openai');
  
  if (!lessonPath) {
    console.error('Usage: node generate-audio.js <lesson.json> [--age-group <2-5|6-12|...>] [--openai]');
    process.exit(1);
  }
  
  generateLessonAudio(lessonPath, ageGroup, useOpenAI).then(success => {
    process.exit(success ? 0 : 1);
  });
}

module.exports = { generateLessonAudio };








