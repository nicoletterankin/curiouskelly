import { NextRequest, NextResponse } from 'next/server'

/**
 * DAY 21 HEYGEN BATCH GENERATION
 * 
 * Generates all 1080 Day 21 lesson videos:
 * 6 languages × 3 ages × 12 archetypes × 5 phases
 * 
 * POST /api/heygen/day21-batch
 * - Starts batch generation
 * - Returns job ID for tracking
 * 
 * GET /api/heygen/day21-batch
 * - Returns current progress
 */

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!

// Kelly voice IDs by age
const VOICES: Record<string, string> = {
  kid: '3WLAUtAsXlPNi4dw1tDM',
  adult: 'BbuMXx40WT4ZuAgRXvNx',
  senior: 'RLUgsoI4JTB52m0yY4yM',
}

// Kelly talking photo IDs by age × archetype
const LOOKS: Record<string, string> = {
  'kid-storyteller': '1024bc304a1146998bc4c360173b2c48',
  'kid-explorer': 'fa4a6780e25a49699ee4f75cb1f03103',
  'kid-scientist': '82813816115c4fbe93b3f3f211bd9931',
  'kid-architect': 'cc1dd0e9e2fd432099985c9b036ed836',
  'kid-strategist': '6249632f58ce479891de00b4da5fb88d',
  'kid-diplomat': '48bddc41ae94473caa645ce9ab93136d',
  'kid-mystic': '5cff601bfb344015a65ff46c6b8cd70a',
  'kid-rebel': 'd4e960f7a3424d869877f3a951adfae7',
  'kid-macgyver': '7b6ab196f2c7430b945411df51a84c58',
  'kid-empath': 'deeb27f2648848b48c5c1ce59059bd54',
  'kid-provider': 'deaa213342944dc2bf671abe1442e316',
  'kid-survivor': 'bd579e4ca77444aca2bfea8ee9070830',
  'adult-storyteller': '3d6a9d6f91b444469dae87ebb3d9eba6',
  'adult-explorer': '62516885ca4b4eae8f63b87b8c060e25',
  'adult-scientist': '277aba5b86a14ff2a4eca2eab2402ab3',
  'adult-architect': '35d0115505824e3182eb9d2ee8cfe73d',
  'adult-strategist': '08d53d1b065041bda2e5b6bc32962a8a',
  'adult-diplomat': 'c3cdbe48fe274420a7f45a4da7e366aa',
  'adult-mystic': 'dfaf9fbd644a475595b178f0be65a39a',
  'adult-rebel': '390be3fb2b064883bb2304fc3968fd87',
  'adult-macgyver': 'c5aab6ab13d940f8ae4700d546bd6b6b',
  'adult-empath': '6bb1a05678c64213a1ed3a4dc790b81e',
  'adult-provider': '9a143feeb2994989b034cebeb78753be',
  'adult-survivor': '831c8d6048104ba0b03a74a36543cfb9',
  'senior-storyteller': '98178c87897e4421884b535b7864ba86',
  'senior-explorer': 'c38e30f2a3cf4e81b0365abf41579f22',
  'senior-scientist': '97e1c9dc1ed04e8fa357c69bde34e58e',
  'senior-architect': '42e9197ab9d84961915b00d5cc780190',
  'senior-strategist': 'e4ab0d4d1f1b4dc9b81a1076b018557f',
  'senior-diplomat': 'a82183881e284e3782db75b755c3f080',
  'senior-mystic': 'c6d104b2ca354b0a9593cb840988bf6e',
  'senior-rebel': 'dc835263eaa247f5b0e06106b848df18',
  'senior-macgyver': 'cb5b025506284d64b696e296ca2feead',
  'senior-empath': '493dac2cf2ba4509b3cc048ff819765e',
  'senior-provider': '12582467e9ff48889d7b2435642e2d65',
  'senior-survivor': '9a143feeb2994989b034cebeb78753be',
}

// Day 21 scripts by language
const DAY21_SCRIPTS: Record<string, Record<string, string>> = {
  en: {
    hook: "Have you ever planted a seed and watched it become something amazing? Today we discover the magic of growth!",
    story: "Every living thing grows. What makes things grow? Food, water, and light. Plants drink through roots and turn sunlight into energy.",
    wonder: "Did you know bamboo can grow three feet in just one day? Taller than most kids!",
    action: "Find a plant nearby. Can you see new leaves starting? Check on it every day this week!",
    wisdom: "Everything that grows is uncomfortable first. A seed must break open. Growth takes effort, but it is worth it.",
  },
  es: {
    hook: "Alguna vez plantaste una semilla y la viste convertirse en algo increible? Hoy descubrimos la magia del crecimiento!",
    story: "Todo ser vivo crece. Que hace crecer las cosas? Comida, agua y luz. Las plantas beben por las raices y convierten la luz solar en energia.",
    wonder: "Sabias que el bambu puede crecer un metro en un solo dia? Mas alto que la mayoria de los ninos!",
    action: "Encuentra una planta cerca. Puedes ver hojas nuevas? Revisala todos los dias esta semana!",
    wisdom: "Todo lo que crece primero es incomodo. Una semilla debe abrirse. El crecimiento requiere esfuerzo, pero vale la pena.",
  },
  fr: {
    hook: "As-tu deja plante une graine et regarde elle devenir quelque chose d'incroyable? Aujourd'hui nous decouvrons la magie de la croissance!",
    story: "Tout etre vivant grandit. Qu'est-ce qui fait grandir? La nourriture, l'eau et la lumiere. Les plantes boivent par les racines.",
    wonder: "Savais-tu que le bambou peut pousser d'un metre en une seule journee? Plus grand que la plupart des enfants!",
    action: "Trouve une plante pres de toi. Vois-tu de nouvelles feuilles? Verifie-la chaque jour cette semaine!",
    wisdom: "Tout ce qui grandit est d'abord inconfortable. Une graine doit s'ouvrir. La croissance demande des efforts.",
  },
  de: {
    hook: "Hast du jemals einen Samen gepflanzt und zugesehen wie er zu etwas Erstaunlichem wurde? Heute entdecken wir die Magie des Wachstums!",
    story: "Jedes Lebewesen waechst. Was laesst Dinge wachsen? Nahrung, Wasser und Licht. Pflanzen trinken durch Wurzeln.",
    wonder: "Wusstest du dass Bambus an einem Tag einen Meter wachsen kann? Groesser als die meisten Kinder!",
    action: "Finde eine Pflanze in der Naehe. Siehst du neue Blaetter? Schau diese Woche jeden Tag nach ihr!",
    wisdom: "Alles was waechst ist zuerst unbequem. Ein Samen muss aufbrechen. Wachstum erfordert Anstrengung.",
  },
  pt: {
    hook: "Voce ja plantou uma semente e viu ela se tornar algo incrivel? Hoje descobrimos a magia do crescimento!",
    story: "Todo ser vivo cresce. O que faz as coisas crescerem? Comida, agua e luz. Plantas bebem pelas raizes.",
    wonder: "Voce sabia que o bambu pode crescer um metro em um unico dia? Mais alto que a maioria das criancas!",
    action: "Encontre uma planta perto de voce. Consegue ver folhas novas? Verifique todos os dias esta semana!",
    wisdom: "Tudo que cresce primeiro e desconfortavel. Uma semente precisa se abrir. Crescimento requer esforco.",
  },
  zh: {
    hook: "Have you ever planted a seed and watched it grow? Today we discover the magic of growth!",
    story: "Every living thing grows. Food, water, and light make things grow. Plants drink through roots.",
    wonder: "Did you know bamboo can grow one meter in just one day? Taller than most kids!",
    action: "Find a plant nearby. Can you see new leaves? Check on it every day this week!",
    wisdom: "Everything that grows is uncomfortable first. A seed must break open. Growth takes effort.",
  },
}

const LANGUAGES = ['en', 'es', 'fr', 'de', 'pt', 'zh']
const AGES = ['kid', 'adult', 'senior']
const ARCHETYPES = ['storyteller', 'explorer', 'scientist', 'architect', 'strategist', 'diplomat', 'mystic', 'rebel', 'macgyver', 'empath', 'provider', 'survivor']
const PHASES = ['hook', 'story', 'wonder', 'action', 'wisdom']

// Check remaining HeyGen credits
async function checkCredits(): Promise<number> {
  const res = await fetch('https://api.heygen.com/v1/user/remaining_quota', {
    headers: { 'X-Api-Key': HEYGEN_API_KEY },
  })
  const data = await res.json()
  return data?.data?.remaining_quota || 0
}

// Generate a single video
async function generateVideo(params: {
  text: string
  voiceId: string
  photoId: string
  title: string
}): Promise<{ videoId: string | null; error: string | null }> {
  try {
    const res = await fetch('https://api.heygen.com/v2/video/generate', {
      method: 'POST',
      headers: {
        'X-Api-Key': HEYGEN_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        video_inputs: [{
          character: { type: 'talking_photo', talking_photo_id: params.photoId },
          voice: { type: 'text', input_text: params.text, voice_id: params.voiceId },
        }],
        dimension: { width: 1920, height: 1080 },
        title: params.title,
      }),
    })
    
    const data = await res.json()
    if (data?.data?.video_id) {
      return { videoId: data.data.video_id, error: null }
    }
    return { videoId: null, error: data?.error?.message || 'Unknown error' }
  } catch (err) {
    return { videoId: null, error: String(err) }
  }
}

export async function GET() {
  // Check credits
  const credits = await checkCredits()
  
  // Calculate total needed
  const total = LANGUAGES.length * AGES.length * ARCHETYPES.length * PHASES.length
  
  return NextResponse.json({
    status: 'ready',
    creditsRemaining: credits,
    totalVideosNeeded: total,
    breakdown: {
      languages: LANGUAGES.length,
      ages: AGES.length,
      archetypes: ARCHETYPES.length,
      phases: PHASES.length,
    },
    message: credits >= total 
      ? `Ready to generate ${total} videos with ${credits} credits`
      : `Need ${total} credits but only have ${credits}`,
  })
}

export async function POST(request: NextRequest) {
  const { startFrom = 0, limit = 10 } = await request.json().catch(() => ({}))
  
  // Check credits first
  const credits = await checkCredits()
  if (credits < limit) {
    return NextResponse.json({ error: `Only ${credits} credits remaining` }, { status: 400 })
  }
  
  const results: { success: string[]; failed: string[]; pending: string[] } = {
    success: [],
    failed: [],
    pending: [],
  }
  
  let index = 0
  
  for (const lang of LANGUAGES) {
    for (const age of AGES) {
      for (const archetype of ARCHETYPES) {
        for (const phase of PHASES) {
          if (index < startFrom) {
            index++
            continue
          }
          
          if (results.success.length + results.pending.length >= limit) {
            return NextResponse.json({
              status: 'partial',
              results,
              nextStartFrom: index,
              creditsUsed: results.success.length + results.pending.length,
            })
          }
          
          const key = `${lang}-${age}-${archetype}-${phase}`
          const script = DAY21_SCRIPTS[lang]?.[phase] || DAY21_SCRIPTS.en[phase]
          const voiceId = VOICES[age]
          const photoId = LOOKS[`${age}-${archetype}`]
          
          if (!photoId) {
            results.failed.push(`${key}: Missing photo ID`)
            index++
            continue
          }
          
          const { videoId, error } = await generateVideo({
            text: script,
            voiceId,
            photoId,
            title: `Day21-${key}`,
          })
          
          if (videoId) {
            results.pending.push(`${key}: ${videoId}`)
          } else {
            results.failed.push(`${key}: ${error}`)
          }
          
          // Rate limit - 2 second delay between API calls
          await new Promise(resolve => setTimeout(resolve, 2000))
          
          index++
        }
      }
    }
  }
  
  return NextResponse.json({
    status: 'complete',
    results,
    creditsUsed: results.success.length + results.pending.length,
  })
}
