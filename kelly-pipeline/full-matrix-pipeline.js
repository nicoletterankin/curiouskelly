/**
 * FULL MATRIX PIPELINE - Day 20 Video Generation
 * 
 * Matrix: 3 ages × 6 languages × 5 phases = 90 videos per engine
 * 
 * Ages: child, adult, elder
 * Languages: en, es, fr, de, pt, cn
 * Phases: hook, story, wonder, action, wisdom
 */

const { createClient } = require('@supabase/supabase-js');
const fetch = require('node-fetch');
require('dotenv').config({ path: 'C:/Users/user/ANTIGRAVITY/media_keys.env' });

// Config
const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const ELEVENLABS_KEY = process.env.ELEVENLABS_API_KEY;
const HEYGEN_KEY = process.env.HEYGEN_API_KEY;
const BLOB_TOKEN = process.env.BLOB_READ_WRITE_TOKEN;

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// Matrix dimensions
const AGES = ['child', 'adult', 'elder'];
const LANGUAGES = ['en', 'es', 'fr', 'de', 'pt', 'cn'];
const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'];
const DAY = 20;

// ElevenLabs voice settings per age
const VOICE_SETTINGS = {
  child: { stability: 0.5, similarity_boost: 0.85, style: 0.3 },
  adult: { stability: 0.75, similarity_boost: 0.75, style: 0.0 },
  elder: { stability: 0.9, similarity_boost: 0.65, style: 0.0 }
};

// Day 20 scripts per language
const SCRIPTS = {
  en: {
    hook: "You learned you have five senses: sight, sound, smell, taste, touch. But that's not quite right. Scientists now count at least 21 distinct senses—and some say more. Today, we discover the senses you didn't know you had.",
    story: "Close your eyes and touch your nose. You did that without looking. How? Not with any of the classic five senses. You used something else—a sense that has a name and works constantly without you noticing.",
    wonder: "Proprioception tells you where your limbs are. Thermoception detects temperature. Nociception signals pain. Equilibrioception keeps you balanced. Chronoception tracks time. Each is as real as sight or sound—just less famous.",
    action: "You've heard we can smell 10,000 odors. That number was guessed in 1927. Modern research shows your nose can distinguish over 1 trillion different scent combinations. Your sense of smell is vastly underrated.",
    wisdom: "Here's today's wisdom: Your senses deliver far more information than you consciously process. Slowing down to really see, really taste, really feel—this is how we wake up to the richness that's always already there."
  },
  es: {
    hook: "Aprendiste que tienes cinco sentidos: vista, oído, olfato, gusto, tacto. Pero eso no es del todo correcto. Los científicos ahora cuentan al menos 21 sentidos distintos—y algunos dicen más. Hoy descubrimos los sentidos que no sabías que tenías.",
    story: "Cierra los ojos y toca tu nariz. Lo hiciste sin mirar. ¿Cómo? No con ninguno de los cinco sentidos clásicos. Usaste algo más—un sentido que tiene un nombre y funciona constantemente sin que lo notes.",
    wonder: "La propiocepción te dice dónde están tus extremidades. La termocepción detecta temperatura. La nocicepción señala dolor. El equilibriocepción te mantiene equilibrado. La cronocepción rastrea el tiempo. Cada uno es tan real como la vista o el sonido—solo menos famoso.",
    action: "Has oído que podemos oler 10,000 olores. Ese número se adivinó en 1927. La investigación moderna muestra que tu nariz puede distinguir más de 1 billón de combinaciones de olores diferentes. Tu sentido del olfato está muy subestimado.",
    wisdom: "Esta es la sabiduría de hoy: Tus sentidos entregan mucha más información de la que procesas conscientemente. Desacelerar para realmente ver, realmente saborear, realmente sentir—así es como despertamos a la riqueza que siempre ya está ahí."
  },
  pt: {
    hook: "Você aprendeu que tem cinco sentidos: visão, audição, olfato, paladar, tato. Mas isso não está bem certo. Cientistas agora contam pelo menos 21 sentidos distintos—e alguns dizem mais. Hoje descobrimos os sentidos que você não sabia que tinha.",
    story: "Feche os olhos e toque seu nariz. Você fez isso sem olhar. Como? Não com nenhum dos cinco sentidos clássicos. Você usou outra coisa—um sentido que tem um nome e funciona constantemente sem você perceber.",
    wonder: "A propriocepção te diz onde estão seus membros. A termocepção detecta temperatura. A nocicepção sinaliza dor. O equilibriocepção te mantém equilibrado. A cronocepção rastreia o tempo. Cada um é tão real quanto visão ou audição—apenas menos famoso.",
    action: "Você já ouviu que podemos cheirar 10.000 odores. Esse número foi adivinhado em 1927. Pesquisas modernas mostram que seu nariz pode distinguir mais de 1 trilhão de combinações de cheiros diferentes. Seu sentido do olfato é muito subestimado.",
    wisdom: "Eis a sabedoria de hoje: Seus sentidos entregam muito mais informação do que você processa conscientemente. Desacelerar para realmente ver, realmente saborear, realmente sentir—é assim que despertamos para a riqueza que sempre já está lá."
  },
  fr: {
    hook: "Vous avez appris que vous avez cinq sens: la vue, l'ouïe, l'odorat, le goût, le toucher. Mais ce n'est pas tout à fait exact. Les scientifiques comptent maintenant au moins 21 sens distincts—et certains en comptent plus. Aujourd'hui, nous découvrons les sens que vous ne saviez pas avoir.",
    story: "Fermez les yeux et touchez votre nez. Vous l'avez fait sans regarder. Comment? Pas avec l'un des cinq sens classiques. Vous avez utilisé autre chose—un sens qui a un nom et fonctionne constamment sans que vous le remarquiez.",
    wonder: "La proprioception vous dit où sont vos membres. La thermoception détecte la température. La nociception signale la douleur. L'équilibrioception vous maintient en équilibre. La chronoception suit le temps. Chacun est aussi réel que la vue ou l'ouïe—juste moins célèbre.",
    action: "Vous avez entendu que nous pouvons sentir 10 000 odeurs. Ce nombre a été deviné en 1927. La recherche moderne montre que votre nez peut distinguer plus d'un trillion de combinaisons d'odeurs différentes. Votre sens de l'odorat est très sous-estimé.",
    wisdom: "Voici la sagesse d'aujourd'hui: Vos sens livrent bien plus d'informations que vous ne traitez consciemment. Ralentir pour vraiment voir, vraiment goûter, vraiment sentir—c'est ainsi que nous nous éveillons à la richesse qui est toujours déjà là."
  },
  de: {
    hook: "Sie haben gelernt, dass Sie fünf Sinne haben: Sehen, Hören, Riechen, Schmecken, Tasten. Aber das stimmt nicht ganz. Wissenschaftler zählen jetzt mindestens 21 verschiedene Sinne—und manche sagen mehr. Heute entdecken wir die Sinne, von denen Sie nicht wussten, dass Sie sie haben.",
    story: "Schließen Sie die Augen und berühren Sie Ihre Nase. Sie haben das getan, ohne hinzusehen. Wie? Nicht mit einem der klassischen fünf Sinne. Sie haben etwas anderes benutzt—einen Sinn, der einen Namen hat und ständig arbeitet, ohne dass Sie es bemerken.",
    wonder: "Die Propriozeption sagt Ihnen, wo Ihre Gliedmaßen sind. Die Thermozeption erkennt Temperatur. Die Nozizeption signalisiert Schmerz. Die Gleichgewichtswahrnehmung hält Sie ausbalanciert. Die Chronozeption verfolgt die Zeit. Jeder ist so real wie Sehen oder Hören—nur weniger berühmt.",
    action: "Sie haben gehört, dass wir 10.000 Gerüche riechen können. Diese Zahl wurde 1927 geschätzt. Moderne Forschung zeigt, dass Ihre Nase über eine Billion verschiedene Geruchskombinationen unterscheiden kann. Ihr Geruchssinn ist stark unterschätzt.",
    wisdom: "Hier ist die Weisheit von heute: Ihre Sinne liefern weit mehr Informationen, als Sie bewusst verarbeiten. Langsamer werden, um wirklich zu sehen, wirklich zu schmecken, wirklich zu fühlen—so erwachen wir zum Reichtum, der immer schon da ist."
  },
  cn: {
    hook: "你学到了你有五种感官：视觉、听觉、嗅觉、味觉、触觉。但这并不完全正确。科学家现在至少计算出21种不同的感官——有些人说更多。今天，我们发现你不知道自己拥有的感官。",
    story: "闭上眼睛，摸你的鼻子。你在不看的情况下做到了。怎么做到的？不是用任何经典的五种感官。你使用了其他东西——一种有名字并且在你没有注意到的情况下不断工作的感官。",
    wonder: "本体感觉告诉你四肢在哪里。温度感知检测温度。伤害感知发出疼痛信号。平衡感让你保持平衡。时间感追踪时间。每一种都像视觉或听觉一样真实——只是不那么有名。",
    action: "你听说我们能闻到一万种气味。那个数字是1927年猜测的。现代研究表明，你的鼻子可以区分超过一万亿种不同的气味组合。你的嗅觉被大大低估了。",
    wisdom: "这是今天的智慧：你的感官传递的信息远比你有意识处理的多得多。慢下来，真正地看、真正地品尝、真正地感受——这就是我们如何觉醒到一直存在的丰富。"
  }
};

// CORRECT Kelly Avatar - Public avatar matching description (real woman, blue shirt)
// Kelly_Blue_Shirt_Front is the standard HeyGen Kelly avatar
const KELLY_AVATAR_ID = 'Kelly_Blue_Shirt_Front';

async function generateAudio(text, language, age, phase) {
  const voiceId = 'wAdymQH5YucAkXwmrdL0'; // Kelly's voice
  
  const resp = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`, {
    method: 'POST',
    headers: {
      'xi-api-key': ELEVENLABS_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      text: text,
      model_id: 'eleven_multilingual_v2',
      voice_settings: VOICE_SETTINGS[age]
    })
  });
  
  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`ElevenLabs error: ${err}`);
  }
  
  return Buffer.from(await resp.arrayBuffer());
}

async function uploadToBlob(buffer, blobPath) {
  const resp = await fetch(`https://blob.vercel-storage.com/${blobPath}`, {
    method: 'PUT',
    headers: {
      'Authorization': `Bearer ${BLOB_TOKEN}`,
      'Content-Type': 'audio/mpeg',
      'x-content-type-options': 'nosniff'
    },
    body: buffer
  });
  
  if (!resp.ok) {
    const text = await resp.text();
    // Check if already exists
    if (text.includes('already exists')) {
      // Construct URL
      return `https://fngrggkpvkplcmmbuwec.public.blob.vercel-storage.com/${blobPath}`;
    }
    throw new Error(`Blob upload error: ${text}`);
  }
  
  const data = await resp.json();
  return data.url;
}

async function createVideoJob(day, language, age, phase, audioUrl, engine = 'heygen') {
  const { data, error } = await supabase
    .from('video_jobs')
    .insert({
      day_of_year: day,
      language: language,
      age_category: age,
      phase: phase,
      engine: engine,
      status: 'queued',
      priority: 10,
      input_payload: {
        audio_url: audioUrl,
        avatar_id: KELLY_AVATAR_ID,  // CORRECT Kelly avatar
        avatar_type: 'avatar'  // NOT talking_photo
      },
      created_at: new Date().toISOString()
    })
    .select()
    .single();
  
  if (error) throw error;
  return data;
}

async function getExistingJobs(day) {
  const { data, error } = await supabase
    .from('video_jobs')
    .select('*')
    .eq('day_of_year', day);
  
  if (error) throw error;
  return data || [];
}

async function submitToHeyGen(audioUrl, language, age, phase) {
  // Use AVATAR type with our custom Kelly avatar - NOT talking_photo!
  const payload = {
    video_inputs: [{
      character: {
        type: 'avatar',
        avatar_id: KELLY_AVATAR_ID,
        avatar_style: 'normal'
      },
      voice: {
        type: 'audio',
        audio_url: audioUrl
      }
    }],
    dimension: {
      width: 1080,
      height: 1920  // Portrait for mobile
    }
  };
  
  const resp = await fetch('https://api.heygen.com/v2/video/generate', {
    method: 'POST',
    headers: {
      'X-Api-Key': HEYGEN_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  });
  
  const data = await resp.json();
  
  if (data.error) {
    throw new Error(`HeyGen error: ${JSON.stringify(data.error)}`);
  }
  
  return data.data?.video_id;
}

async function main() {
  console.log('='.repeat(60));
  console.log('FULL MATRIX PIPELINE - Day 20');
  console.log('='.repeat(60));
  console.log(`Matrix: ${AGES.length} ages x ${LANGUAGES.length} languages x ${PHASES.length} phases`);
  console.log(`Total combinations: ${AGES.length * LANGUAGES.length * PHASES.length}`);
  console.log('');
  
  // Check existing jobs
  const existingJobs = await getExistingJobs(DAY);
  console.log(`Existing jobs for Day ${DAY}: ${existingJobs.length}`);
  
  // Create a set of existing combinations
  const existing = new Set(
    existingJobs.map(j => `${j.language}-${j.age_category}-${j.phase}-${j.engine}`)
  );
  
  let created = 0;
  let skipped = 0;
  let errors = [];
  
  // Process non-EN languages first (EN/adult already done)
  const languageOrder = ['es', 'fr', 'de', 'pt', 'cn', 'en'];
  
  for (const lang of languageOrder) {
    console.log(`\n--- Processing language: ${lang.toUpperCase()} ---`);
    
    if (!SCRIPTS[lang]) {
      console.log(`  WARNING: No scripts for ${lang}, skipping...`);
      continue;
    }
    
    for (const age of AGES) {
      // Skip EN/adult as it's already done
      if (lang === 'en' && age === 'adult') {
        console.log(`  Skipping en/adult (already done)`);
        skipped += 5;
        continue;
      }
      
      for (const phase of PHASES) {
        const key = `${lang}-${age}-${phase}-heygen`;
        
        if (existing.has(key)) {
          console.log(`  Exists: ${lang}/${age}/${phase}`);
          skipped++;
          continue;
        }
        
        try {
          const text = SCRIPTS[lang][phase];
          if (!text) {
            console.log(`  No text for ${phase} in ${lang}`);
            continue;
          }
          
          console.log(`  Creating: ${lang}/${age}/${phase}`);
          
          // Generate audio
          console.log(`    Generating audio...`);
          const audioBuffer = await generateAudio(text, lang, age, phase);
          
          // Upload to Blob
          const blobPath = `audio/2026/${lang}/day-${DAY}/${phase}-${age}.mp3`;
          console.log(`    Uploading to blob...`);
          const audioUrl = await uploadToBlob(audioBuffer, blobPath);
          console.log(`    Audio: ${audioUrl.slice(0, 60)}...`);
          
          // Create job in DB
          const job = await createVideoJob(DAY, lang, age, phase, audioUrl);
          console.log(`    Job ID: ${job.id}`);
          
          // Submit to HeyGen
          console.log(`    Submitting to HeyGen...`);
          const videoId = await submitToHeyGen(audioUrl, lang, age, phase);
          console.log(`    HeyGen video ID: ${videoId}`);
          
          // Update job with external_id
          await supabase
            .from('video_jobs')
            .update({ 
              external_id: videoId, 
              status: 'submitted',
              submitted_at: new Date().toISOString()
            })
            .eq('id', job.id);
          
          created++;
          
          // Rate limit - 1 second between requests
          await new Promise(r => setTimeout(r, 1000));
          
        } catch (err) {
          console.log(`    ERROR: ${err.message}`);
          errors.push({ key: `${lang}/${age}/${phase}`, error: err.message });
        }
      }
    }
  }
  
  console.log('\n' + '='.repeat(60));
  console.log('SUMMARY');
  console.log('='.repeat(60));
  console.log(`Created: ${created}`);
  console.log(`Skipped (existing): ${skipped}`);
  console.log(`Errors: ${errors.length}`);
  
  if (errors.length > 0) {
    console.log('\nErrors:');
    errors.slice(0, 10).forEach(e => console.log(`  ${e.key}: ${e.error}`));
    if (errors.length > 10) {
      console.log(`  ... and ${errors.length - 10} more`);
    }
  }
}

// Check current DB state
async function checkStatus() {
  const { data, error } = await supabase
    .from('video_jobs')
    .select('day_of_year, language, age_category, phase, engine, status')
    .eq('day_of_year', 20)
    .order('created_at', { ascending: false });
  
  if (error) {
    console.error('Error:', error);
    return;
  }
  
  console.log(`Total Day 20 jobs: ${data.length}`);
  
  // Group by status
  const byStatus = {};
  data.forEach(j => {
    byStatus[j.status] = (byStatus[j.status] || 0) + 1;
  });
  console.log('\nBy status:', byStatus);
  
  // Group by language
  const byLang = {};
  data.forEach(j => {
    byLang[j.language] = (byLang[j.language] || 0) + 1;
  });
  console.log('\nBy language:', byLang);
  
  // Group by age
  const byAge = {};
  data.forEach(j => {
    byAge[j.age_category] = (byAge[j.age_category] || 0) + 1;
  });
  console.log('\nBy age:', byAge);
  
  // Group by engine
  const byEngine = {};
  data.forEach(j => {
    byEngine[j.engine] = (byEngine[j.engine] || 0) + 1;
  });
  console.log('\nBy engine:', byEngine);
  
  // What's missing?
  console.log('\n--- COVERAGE ANALYSIS ---');
  const existing = new Set(data.map(j => `${j.language}-${j.age_category}-${j.phase}`));
  const missing = [];
  for (const lang of LANGUAGES) {
    for (const age of AGES) {
      for (const phase of PHASES) {
        const key = `${lang}-${age}-${phase}`;
        if (!existing.has(key)) {
          missing.push(key);
        }
      }
    }
  }
  console.log(`Missing combinations: ${missing.length}`);
  if (missing.length > 0 && missing.length <= 20) {
    console.log('Missing:', missing);
  }
}

// Parse command
const cmd = process.argv[2] || 'status';
if (cmd === 'run') {
  main().catch(console.error);
} else {
  checkStatus().catch(console.error);
}
