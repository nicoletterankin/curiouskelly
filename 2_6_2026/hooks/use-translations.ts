'use client'

import useSWR from 'swr'

interface TranslatedLesson {
  title: string
  subtitle: string | null
  hook_script: string
  story_script: string
  wonder_script: string
  action_script: string
  wisdom_script: string
  quote: string | null
  quote_author: string | null
}

interface TranslationResponse {
  day: number
  language: string
  translation: TranslatedLesson | null
  fallback: boolean
}

const fetcher = async (url: string): Promise<TranslationResponse> => {
  try {
    const res = await fetch(url)
    if (!res.ok) return { day: 0, language: 'en', translation: null, fallback: true }
    return res.json()
  } catch {
    return { day: 0, language: 'en', translation: null, fallback: true }
  }
}

export function useTranslations(day: number, language: string) {
  const { data, error, isLoading } = useSWR<TranslationResponse>(
    language !== 'en' ? `/api/translations?day=${day}&lang=${language}` : null,
    fetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 60000,
    }
  )
  
  return {
    translation: data?.translation || null,
    isFallback: data?.fallback ?? true,
    isLoading,
    error,
  }
}

// UI translations for interface elements
// Locale mapping for date formatting
export const LOCALE_MAP: Record<string, string> = {
  en: 'en-US',
  es: 'es-ES',
  fr: 'fr-FR',
  de: 'de-DE',
  pt: 'pt-BR',
  zh: 'zh-CN',
}

// Format date with proper locale
export function formatLocalizedDate(date: Date, language: string, options: Intl.DateTimeFormatOptions): string {
  const locale = LOCALE_MAP[language] || 'en-US'
  try {
    return date.toLocaleDateString(locale, options)
  } catch {
    return date.toLocaleDateString('en-US', options)
  }
}

export const UI_TRANSLATIONS: Record<string, Record<string, string>> = {
  en: {
    'lets_begin': "Let's Begin",
    'learning_now': 'learning now',
    'learning': 'learning',
    'todaysLesson': "Today's lesson",
    'theDailyLesson': 'The Daily Lesson',
    'next_live_class': 'NEXT LIVE CLASS',
    'with_kelly_starts_in': 'with Kelly starts in',
    'live_comments': 'LIVE COMMENTS',
    'simulated_with_ai': 'Simulated with AI',
    'comment_as': 'Comment as',
    'share_this_lesson': 'Share this lesson',
    'translate': 'Translate',
    'true': 'TRUE',
    'false': 'FALSE',
    'isThis': 'IS THIS',
    'or': 'OR',
    'day_of': 'Day {day} of 365',
    'save': 'Save',
    'play': 'Play',
    'pause': 'Pause',
    'next': 'Next',
    'mute': 'Mute',
    'unmute': 'Unmute',
    'cc': 'CC',
    'more': 'More',
    'correct': 'Correct!',
    'notQuite': 'Not quite!',
    'theAnswerIs': 'The answer is',
    'whyItsTrue': 'Why it\'s true',
    'explanation_hook': 'Growth requires leaving your comfort zone. Every living thing—from plants reaching toward light to humans learning new skills—must push beyond what feels easy. That discomfort is the signal that growth is happening.',
    'saidTrue': 'said True',
    'votes': 'votes',
    'kellysScript': 'Kelly\'s Script',
    'showScript': 'Show script',
    'live': 'Live',
    'kelly': 'Kelly',
    'find': 'Find',
    'settings': 'Settings',
    'like': 'Like',
    'comment': 'Comment',
    'liveClass': 'LIVE CLASS',
    'shareLesson': 'Share this lesson',
    'copyLink': 'Copy Link',
    'copied': 'Copied!',
    'compareEngines': 'Compare Engines',
    'videoFineTuning': 'Video Engine Fine-Tuning',
    'compareLipSync': 'Compare lip-sync quality across engines',
    'tapToHear': 'Tap to hear Kelly',
  },
  es: {
    'lets_begin': 'Comencemos',
    'learning_now': 'aprendiendo ahora',
    'learning': 'aprendiendo',
    'todaysLesson': 'Leccion de hoy',
    'theDailyLesson': 'La Leccion Diaria',
    'next_live_class': 'PROXIMA CLASE EN VIVO',
    'with_kelly_starts_in': 'con Kelly comienza en',
    'live_comments': 'COMENTARIOS EN VIVO',
    'simulated_with_ai': 'Simulado con IA',
    'comment_as': 'Comentar como',
    'share_this_lesson': 'Compartir esta leccion',
    'translate': 'Traducir',
    'true': 'VERDADERO',
    'false': 'FALSO',
    'isThis': 'ES ESTO',
    'or': 'O',
    'day_of': 'Dia {day} de 365',
    'save': 'Guardar',
    'play': 'Reproducir',
    'next': 'Siguiente',
    'mute': 'Silenciar',
    'unmute': 'Activar sonido',
    'cc': 'CC',
    'more': 'Mas',
    'correct': 'Correcto!',
    'notQuite': 'No del todo!',
    'theAnswerIs': 'La respuesta es',
    'whyItsTrue': 'Por que es verdad',
    'explanation_hook': 'El crecimiento requiere salir de tu zona de confort. Todo ser vivo, desde las plantas que buscan la luz hasta los humanos que aprenden nuevas habilidades, debe ir mas alla de lo que se siente facil. Esa incomodidad es la senal de que el crecimiento esta ocurriendo.',
    'saidTrue': 'dijeron Verdadero',
    'votes': 'votos',
    'kellysScript': 'Guion de Kelly',
    'showScript': 'Mostrar guion',
    'live': 'En vivo',
    'kelly': 'Kelly',
    'find': 'Buscar',
    'settings': 'Ajustes',
    'like': 'Me gusta',
    'comment': 'Comentar',
    'pause': 'Pausa',
    'liveClass': 'CLASE EN VIVO',
    'shareLesson': 'Compartir leccion',
    'copyLink': 'Copiar enlace',
    'copied': 'Copiado!',
    'compareEngines': 'Comparar motores',
'videoFineTuning': 'Ajuste de motor de video',
  'compareLipSync': 'Comparar sincronizacion labial',
  'tapToHear': 'Toca para escuchar a Kelly',
  },
  fr: {
    'lets_begin': 'Commencons',
    'learning_now': 'en apprentissage',
    'learning': 'apprennent',
    'todaysLesson': 'Lecon du jour',
    'theDailyLesson': 'La Lecon Quotidienne',
    'next_live_class': 'PROCHAINE CLASSE EN DIRECT',
    'with_kelly_starts_in': 'avec Kelly commence dans',
    'live_comments': 'COMMENTAIRES EN DIRECT',
    'simulated_with_ai': 'Simule avec IA',
    'comment_as': 'Commenter en tant que',
    'share_this_lesson': 'Partager cette lecon',
    'translate': 'Traduire',
    'true': 'VRAI',
    'false': 'FAUX',
    'isThis': 'EST-CE',
    'or': 'OU',
    'day_of': 'Jour {day} sur 365',
    'save': 'Enregistrer',
    'play': 'Lecture',
    'next': 'Suivant',
    'mute': 'Muet',
    'unmute': 'Activer le son',
    'cc': 'CC',
    'more': 'Plus',
    'correct': 'Correct!',
    'notQuite': 'Pas tout a fait!',
    'theAnswerIs': 'La reponse est',
    'whyItsTrue': 'Pourquoi c\'est vrai',
    'explanation_hook': 'La croissance necessite de sortir de sa zone de confort. Chaque etre vivant, des plantes vers la lumiere aux humains apprenant de nouvelles competences, doit aller au-dela de ce qui semble facile. Cet inconfort est le signe que la croissance se produit.',
    'saidTrue': 'ont dit Vrai',
    'votes': 'votes',
    'kellysScript': 'Script de Kelly',
    'showScript': 'Afficher le script',
    'live': 'En direct',
    'kelly': 'Kelly',
    'find': 'Chercher',
    'settings': 'Parametres',
    'like': 'Aimer',
    'comment': 'Commenter',
    'pause': 'Pause',
    'liveClass': 'COURS EN DIRECT',
    'shareLesson': 'Partager la lecon',
    'copyLink': 'Copier le lien',
    'copied': 'Copie!',
    'compareEngines': 'Comparer les moteurs',
'videoFineTuning': 'Reglage du moteur video',
  'compareLipSync': 'Comparer la synchronisation labiale',
  'tapToHear': 'Appuyez pour ecouter Kelly',
  },
  de: {
    'lets_begin': 'Lass uns beginnen',
    'learning_now': 'lernen gerade',
    'learning': 'lernen',
    'todaysLesson': 'Heutige Lektion',
    'theDailyLesson': 'Die Tagliche Lektion',
    'next_live_class': 'NACHSTE LIVE-KLASSE',
    'with_kelly_starts_in': 'mit Kelly beginnt in',
    'live_comments': 'LIVE-KOMMENTARE',
    'simulated_with_ai': 'Simuliert mit KI',
    'comment_as': 'Kommentieren als',
    'share_this_lesson': 'Diese Lektion teilen',
    'translate': 'Ubersetzen',
    'true': 'WAHR',
    'false': 'FALSCH',
    'isThis': 'IST DAS',
    'or': 'ODER',
    'day_of': 'Tag {day} von 365',
    'save': 'Speichern',
    'play': 'Abspielen',
    'next': 'Weiter',
    'mute': 'Stumm',
    'unmute': 'Ton aktivieren',
    'cc': 'CC',
    'more': 'Mehr',
    'correct': 'Richtig!',
    'notQuite': 'Nicht ganz!',
    'theAnswerIs': 'Die Antwort ist',
    'whyItsTrue': 'Warum es wahr ist',
    'explanation_hook': 'Wachstum erfordert, die Komfortzone zu verlassen. Jedes Lebewesen, von Pflanzen, die zum Licht streben, bis zu Menschen, die neue Fahigkeiten erlernen, muss uber das Einfache hinausgehen. Dieses Unbehagen ist das Zeichen, dass Wachstum stattfindet.',
    'saidTrue': 'sagten Wahr',
    'votes': 'Stimmen',
    'kellysScript': 'Kellys Skript',
    'showScript': 'Skript anzeigen',
    'live': 'Live',
    'kelly': 'Kelly',
    'find': 'Suchen',
    'settings': 'Einstellungen',
    'like': 'Gefallt mir',
    'comment': 'Kommentieren',
    'pause': 'Pause',
    'liveClass': 'LIVE-KLASSE',
    'shareLesson': 'Lektion teilen',
    'copyLink': 'Link kopieren',
    'copied': 'Kopiert!',
    'compareEngines': 'Motoren vergleichen',
'videoFineTuning': 'Video-Engine-Feinabstimmung',
  'compareLipSync': 'Lippensynchronisation vergleichen',
  'tapToHear': 'Tippen um Kelly zu horen',
  },
  pt: {
    'lets_begin': 'Vamos comecar',
    'learning_now': 'aprendendo agora',
    'learning': 'aprendendo',
    'todaysLesson': 'Licao de hoje',
    'theDailyLesson': 'A Licao Diaria',
    'next_live_class': 'PROXIMA AULA AO VIVO',
    'with_kelly_starts_in': 'com Kelly comeca em',
    'live_comments': 'COMENTARIOS AO VIVO',
    'simulated_with_ai': 'Simulado com IA',
    'comment_as': 'Comentar como',
    'share_this_lesson': 'Compartilhar esta licao',
    'translate': 'Traduzir',
    'true': 'VERDADEIRO',
    'false': 'FALSO',
    'isThis': 'ISSO E',
    'or': 'OU',
    'day_of': 'Dia {day} de 365',
    'save': 'Salvar',
    'play': 'Reproduzir',
    'next': 'Proximo',
    'mute': 'Mudo',
    'unmute': 'Ativar som',
    'cc': 'CC',
    'more': 'Mais',
    'correct': 'Correto!',
    'notQuite': 'Nao exatamente!',
    'theAnswerIs': 'A resposta e',
    'whyItsTrue': 'Por que e verdade',
    'explanation_hook': 'O crescimento requer sair da zona de conforto. Todo ser vivo, desde plantas buscando luz ate humanos aprendendo novas habilidades, deve ir alem do que parece facil. Esse desconforto e o sinal de que o crescimento esta acontecendo.',
    'saidTrue': 'disseram Verdadeiro',
    'votes': 'votos',
    'kellysScript': 'Roteiro da Kelly',
    'showScript': 'Mostrar roteiro',
    'live': 'Ao vivo',
    'kelly': 'Kelly',
    'find': 'Buscar',
    'settings': 'Configuracoes',
    'like': 'Curtir',
    'comment': 'Comentar',
    'pause': 'Pausa',
    'liveClass': 'AULA AO VIVO',
    'shareLesson': 'Compartilhar licao',
    'copyLink': 'Copiar link',
    'copied': 'Copiado!',
    'compareEngines': 'Comparar motores',
'videoFineTuning': 'Ajuste do motor de video',
  'compareLipSync': 'Comparar sincronizacao labial',
  'tapToHear': 'Toque para ouvir Kelly',
  },
  zh: {
    'lets_begin': '开始吧',
    'learning_now': '正在学习',
    'learning': '学习中',
    'todaysLesson': '今日课程',
    'theDailyLesson': '每日一课',
    'next_live_class': '下一节直播课',
    'with_kelly_starts_in': 'Kelly的课程开始于',
    'live_comments': '实时评论',
    'simulated_with_ai': 'AI模拟',
    'comment_as': '评论身份',
    'share_this_lesson': '分享这节课',
    'translate': '翻译',
    'true': '正确',
    'false': '错误',
    'isThis': '这是',
    'or': '还是',
    'day_of': '第{day}天/共365天',
    'save': '保存',
    'play': '播放',
    'next': '下一个',
    'mute': '静音',
    'unmute': '取消静音',
    'cc': '字幕',
    'more': '更多',
    'correct': '正确！',
    'notQuite': '不太对！',
    'theAnswerIs': '答案是',
    'whyItsTrue': '为什么是真的',
    'explanation_hook': '成长需要离开舒适区。每个生物，从寻找光线的植物到学习新技能的人类，都必须超越简单的事物。这种不适是成长正在发生的信号。',
    'saidTrue': '选择了正确',
    'votes': '票',
    'kellysScript': 'Kelly的剧本',
    'showScript': '显示剧本',
    'live': '直播',
    'kelly': 'Kelly',
    'find': '搜索',
    'settings': '设置',
    'like': '点赞',
    'comment': '评论',
    'pause': '暂停',
    'liveClass': '直播课程',
    'shareLesson': '分享课程',
    'copyLink': '复制链接',
    'copied': '已复制!',
    'compareEngines': '比较引擎',
'videoFineTuning': '视频引擎调优',
  'compareLipSync': '比较口型同步质量',
  'tapToHear': '点击聆听Kelly',
  },
  }

export function useUITranslation(language: string) {
  const translations = UI_TRANSLATIONS[language] || UI_TRANSLATIONS['en']
  
  const t = (key: string, params?: Record<string, string | number>): string => {
    let text = translations[key] || UI_TRANSLATIONS['en'][key] || key
    if (params) {
      Object.entries(params).forEach(([k, v]) => {
        text = text.replace(`{${k}}`, String(v))
      })
    }
    return text
  }
  
  return { t }
}
