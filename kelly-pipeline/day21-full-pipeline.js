/**
 * DAY 21 - Full Pipeline: Audio Generation + HeyGen Video Submission
 * Topic: "The Stars" - Every atom in your body was forged inside a star
 * Matrix: 3 ages × 6 languages × 5 phases = 90 videos
 */

const { createClient } = require('@supabase/supabase-js');
const fetch = require('node-fetch');
const { put } = require('@vercel/blob');
const { getKellyAvatar } = require('./kelly-avatars');
require('dotenv').config({ path: 'C:/Users/user/ANTIGRAVITY/media_keys.env' });

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const HEYGEN_KEY = process.env.HEYGEN_API_KEY;
const ELEVENLABS_KEY = process.env.ELEVENLABS_API_KEY;
const BLOB_TOKEN = process.env.BLOB_READ_WRITE_TOKEN;

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// Day 21 Scripts - "The Stars"
const DAY21_SCRIPTS = {
  en: {
    hook: "Today we're discovering something magical about The Stars. Look up at the night sky and know this: you are literally made of stars. Every atom in your body was forged in the heart of an ancient star billions of years ago.",
    story: "Let's start with the basics. The first thing to understand about the stars is that they affect our daily lives in ways we often don't notice. The iron in your blood, the calcium in your bones, the oxygen you breathe - all created in stellar explosions.",
    wonder: "Building on that: Scientists have discovered fascinating patterns in how the stars work in nature. When massive stars die in supernovas, they scatter their elements across the cosmos, seeding new planets and eventually, new life.",
    action: "And here's the practical application. Every atom in your body was forged inside a star - and this principle connects to so many other areas of knowledge. You are the universe experiencing itself.",
    wisdom: "The takeaway: Every atom in your body was forged inside a star. Knowledge that makes a difference. We are all stardust, connected to the cosmos in the most fundamental way possible."
  },
  es: {
    hook: "Hoy descubriremos algo mágico sobre Las Estrellas. Mira al cielo nocturno y sabe esto: estás literalmente hecho de estrellas. Cada átomo de tu cuerpo fue forjado en el corazón de una estrella antigua hace miles de millones de años.",
    story: "Comencemos con lo básico. Lo primero que hay que entender sobre las estrellas es que afectan nuestra vida diaria de formas que a menudo no notamos. El hierro en tu sangre, el calcio en tus huesos, el oxígeno que respiras - todo creado en explosiones estelares.",
    wonder: "Profundizando en eso: Los científicos han descubierto patrones fascinantes en cómo funcionan las estrellas en la naturaleza. Cuando las estrellas masivas mueren en supernovas, esparcen sus elementos por el cosmos, sembrando nuevos planetas y eventualmente, nueva vida.",
    action: "Y aquí está la aplicación práctica. Cada átomo de tu cuerpo fue forjado dentro de una estrella - y este principio se conecta con muchas otras áreas del conocimiento. Eres el universo experimentándose a sí mismo.",
    wisdom: "La conclusión: Cada átomo de tu cuerpo fue forjado dentro de una estrella. Conocimiento que marca la diferencia. Todos somos polvo de estrellas, conectados al cosmos de la manera más fundamental posible."
  },
  fr: {
    hook: "Aujourd'hui, nous découvrons quelque chose de magique à propos des Étoiles. Regardez le ciel nocturne et sachez ceci: vous êtes littéralement fait d'étoiles. Chaque atome de votre corps a été forgé au cœur d'une étoile ancienne il y a des milliards d'années.",
    story: "Commençons par les bases. La première chose à comprendre à propos des étoiles est qu'elles affectent notre vie quotidienne de manières que nous ne remarquons souvent pas. Le fer dans votre sang, le calcium dans vos os, l'oxygène que vous respirez - tout créé dans des explosions stellaires.",
    wonder: "En s'appuyant là-dessus: Les scientifiques ont découvert des modèles fascinants dans la façon dont les étoiles fonctionnent dans la nature. Quand les étoiles massives meurent en supernovas, elles dispersent leurs éléments à travers le cosmos, semant de nouvelles planètes et éventuellement, une nouvelle vie.",
    action: "Et voici l'application pratique. Chaque atome de votre corps a été forgé à l'intérieur d'une étoile - et ce principe se connecte à tant d'autres domaines de la connaissance. Vous êtes l'univers qui fait l'expérience de lui-même.",
    wisdom: "Ce qu'il faut retenir: Chaque atome de votre corps a été forgé à l'intérieur d'une étoile. Le savoir qui fait la différence. Nous sommes tous poussière d'étoiles, connectés au cosmos de la manière la plus fondamentale possible."
  },
  de: {
    hook: "Heute entdecken wir etwas Magisches über die Sterne. Schauen Sie zum Nachthimmel und wissen Sie dies: Sie sind buchstäblich aus Sternen gemacht. Jedes Atom in Ihrem Körper wurde im Herzen eines alten Sterns vor Milliarden von Jahren geschmiedet.",
    story: "Beginnen wir mit den Grundlagen. Das Erste, was man über die Sterne verstehen muss, ist, dass sie unser tägliches Leben auf Weisen beeinflussen, die wir oft nicht bemerken. Das Eisen in Ihrem Blut, das Kalzium in Ihren Knochen, der Sauerstoff, den Sie atmen - alles in Sternexplosionen erschaffen.",
    wonder: "Darauf aufbauend: Wissenschaftler haben faszinierende Muster entdeckt, wie die Sterne in der Natur funktionieren. Wenn massive Sterne in Supernovas sterben, verstreuen sie ihre Elemente durch den Kosmos und säen neue Planeten und schließlich neues Leben.",
    action: "Und hier ist die praktische Anwendung. Jedes Atom in Ihrem Körper wurde in einem Stern geschmiedet - und dieses Prinzip verbindet sich mit so vielen anderen Wissensbereichen. Sie sind das Universum, das sich selbst erlebt.",
    wisdom: "Die Erkenntnis: Jedes Atom in Ihrem Körper wurde in einem Stern geschmiedet. Wissen, das einen Unterschied macht. Wir sind alle Sternenstaub, auf die grundlegendste Weise mit dem Kosmos verbunden."
  },
  pt: {
    hook: "Hoje estamos descobrindo algo mágico sobre As Estrelas. Olhe para o céu noturno e saiba disto: você é literalmente feito de estrelas. Cada átomo do seu corpo foi forjado no coração de uma estrela antiga bilhões de anos atrás.",
    story: "Vamos começar com o básico. A primeira coisa a entender sobre as estrelas é que elas afetam nossas vidas diárias de maneiras que muitas vezes não percebemos. O ferro no seu sangue, o cálcio nos seus ossos, o oxigênio que você respira - tudo criado em explosões estelares.",
    wonder: "Construindo sobre isso: Os cientistas descobriram padrões fascinantes em como as estrelas funcionam na natureza. Quando estrelas massivas morrem em supernovas, elas espalham seus elementos pelo cosmos, semeando novos planetas e eventualmente, nova vida.",
    action: "E aqui está a aplicação prática. Cada átomo do seu corpo foi forjado dentro de uma estrela - e este princípio se conecta com tantas outras áreas do conhecimento. Você é o universo experimentando a si mesmo.",
    wisdom: "A conclusão: Cada átomo do seu corpo foi forjado dentro de uma estrela. Conhecimento que faz diferença. Somos todos poeira de estrelas, conectados ao cosmos da maneira mais fundamental possível."
  },
  cn: {
    hook: "今天我们将发现关于星星的神奇之处。仰望夜空，要知道这一点：你实际上是由星星组成的。你身体里的每一个原子都是在数十亿年前古老恒星的核心中锻造的。",
    story: "让我们从基础开始。关于星星首先要了解的是，它们以我们经常没有注意到的方式影响着我们的日常生活。你血液中的铁，骨骼中的钙，你呼吸的氧气 - 都是在恒星爆炸中产生的。",
    wonder: "在此基础上：科学家们发现了星星在自然界中运作的迷人模式。当巨大的恒星在超新星中死亡时，它们将元素散布到宇宙中，播种新的行星，最终孕育新的生命。",
    action: "这里是实际应用。你身体里的每一个原子都是在恒星内部锻造的 - 这个原理与许多其他知识领域相连。你就是宇宙在体验自己。",
    wisdom: "要点：你身体里的每一个原子都是在恒星内部锻造的。有意义的知识。我们都是星尘，以最根本的方式与宇宙相连。"
  }
};

const AGES = ['child', 'adult', 'elder'];
const LANGUAGES = ['en', 'es', 'fr', 'de', 'pt', 'cn'];
const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom'];

// Voice settings per age
const VOICE_SETTINGS = {
  child: { stability: 0.5, similarity_boost: 0.75, style: 0.4 },
  adult: { stability: 0.5, similarity_boost: 0.75, style: 0.3 },
  elder: { stability: 0.6, similarity_boost: 0.70, style: 0.2 }
};

const KELLY_VOICE_ID = 'wAdymQH5YucAkXwmrdL0'; // ElevenLabs Kelly voice

async function generateAudio(text, age, language) {
  const settings = VOICE_SETTINGS[age];
  
  const resp = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${KELLY_VOICE_ID}`, {
    method: 'POST',
    headers: {
      'xi-api-key': ELEVENLABS_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      text: text,
      model_id: 'eleven_multilingual_v2',
      voice_settings: settings
    })
  });
  
  if (!resp.ok) {
    throw new Error(`ElevenLabs error: ${resp.status}`);
  }
  
  return Buffer.from(await resp.arrayBuffer());
}

async function uploadToBlob(buffer, path) {
  const blob = await put(path, buffer, {
    access: 'public',
    token: BLOB_TOKEN
  });
  return blob.url;
}

async function submitToHeyGen(audioUrl, age) {
  const kelly = getKellyAvatar(age, 'storyteller');
  
  const payload = {
    video_inputs: [{
      character: {
        type: 'talking_photo',
        talking_photo_id: kelly.look_id
      },
      voice: {
        type: 'audio',
        audio_url: audioUrl
      }
    }],
    dimension: { width: 1080, height: 1920 }
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
    throw new Error(`HeyGen: ${JSON.stringify(data.error)}`);
  }
  
  return data.data?.video_id;
}

async function processDay21() {
  console.log('='.repeat(60));
  console.log('DAY 21 - THE STARS - Full Pipeline');
  console.log('Matrix: 3 ages × 6 languages × 5 phases = 90 videos');
  console.log('='.repeat(60));
  
  let audioGenerated = 0;
  let jobsCreated = 0;
  let jobsSubmitted = 0;
  let errors = [];
  
  for (const age of AGES) {
    for (const lang of LANGUAGES) {
      for (const phase of PHASES) {
        const key = `${lang}/${age}/${phase}`;
        
        try {
          // Check if job already exists
          const { data: existing } = await supabase
            .from('video_jobs')
            .select('id, status')
            .eq('day_of_year', 21)
            .eq('age_category', age)
            .eq('language', lang)
            .eq('phase', phase)
            .eq('engine', 'heygen')
            .single();
          
          if (existing && existing.status !== 'failed') {
            console.log(`  ${key}: exists (${existing.status})`);
            continue;
          }
          
          console.log(`  ${key}: generating...`);
          
          // Get script
          const script = DAY21_SCRIPTS[lang]?.[phase];
          if (!script) {
            console.log(`    Skip: no script`);
            continue;
          }
          
          // Generate audio
          const audioBuffer = await generateAudio(script, age, lang);
          audioGenerated++;
          
          // Upload to blob
          const audioPath = `audio/2026/${lang}/day-021/${phase}-${age}.mp3`;
          const audioUrl = await uploadToBlob(audioBuffer, audioPath);
          console.log(`    Audio: ${audioUrl.slice(0, 50)}...`);
          
          // Submit to HeyGen
          const videoId = await submitToHeyGen(audioUrl, age);
          console.log(`    HeyGen ID: ${videoId}`);
          jobsSubmitted++;
          
          // Get Kelly avatar info
          const kelly = getKellyAvatar(age);
          
          // Upsert job
          if (existing) {
            await supabase
              .from('video_jobs')
              .update({
                external_id: videoId,
                status: 'submitted',
                submitted_at: new Date().toISOString(),
                input_payload: {
                  audio_url: audioUrl,
                  script: script,
                  talking_photo_id: kelly.look_id,
                  kelly_age_group: kelly.age_group,
                  correct_kelly: true
                }
              })
              .eq('id', existing.id);
          } else {
            await supabase
              .from('video_jobs')
              .insert({
                day_of_year: 21,
                phase: phase,
                age_category: age,
                language: lang,
                engine: 'heygen',
                status: 'submitted',
                external_id: videoId,
                submitted_at: new Date().toISOString(),
                input_payload: {
                  audio_url: audioUrl,
                  script: script,
                  talking_photo_id: kelly.look_id,
                  kelly_age_group: kelly.age_group,
                  correct_kelly: true
                }
              });
            jobsCreated++;
          }
          
          // Rate limit
          await new Promise(r => setTimeout(r, 1500));
          
        } catch (err) {
          console.log(`    ERROR: ${err.message}`);
          errors.push({ key, error: err.message });
        }
      }
    }
  }
  
  console.log('\n' + '='.repeat(60));
  console.log('DAY 21 PIPELINE SUMMARY');
  console.log('='.repeat(60));
  console.log(`Audio generated: ${audioGenerated}`);
  console.log(`Jobs created: ${jobsCreated}`);
  console.log(`Jobs submitted: ${jobsSubmitted}`);
  console.log(`Errors: ${errors.length}`);
  
  if (errors.length > 0) {
    console.log('\nErrors:');
    errors.slice(0, 10).forEach(e => console.log(`  ${e.key}: ${e.error}`));
  }
  
  return { audioGenerated, jobsCreated, jobsSubmitted, errors };
}

async function showStatus() {
  const { data, error } = await supabase
    .from('video_jobs')
    .select('status, language, age_category, phase')
    .eq('day_of_year', 21)
    .eq('engine', 'heygen');
  
  if (error) {
    console.error('Error:', error);
    return;
  }
  
  console.log('='.repeat(60));
  console.log('DAY 21 STATUS');
  console.log('='.repeat(60));
  
  const byStatus = {};
  data.forEach(j => {
    byStatus[j.status] = (byStatus[j.status] || 0) + 1;
  });
  console.log('By status:', byStatus);
  
  const byLang = {};
  data.filter(j => j.status === 'completed').forEach(j => {
    byLang[j.language] = (byLang[j.language] || 0) + 1;
  });
  console.log('Completed by language:', byLang);
  
  const byAge = {};
  data.forEach(j => {
    byAge[j.age_category] = (byAge[j.age_category] || 0) + 1;
  });
  console.log('By age:', byAge);
}

// Main
const cmd = process.argv[2] || 'status';
if (cmd === 'run') {
  processDay21().catch(console.error);
} else {
  showStatus().catch(console.error);
}
