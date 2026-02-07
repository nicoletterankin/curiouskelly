/**
 * UI Translations for Kelly App
 * All user-facing text should use these translations
 */

export type SupportedLanguage = 'en' | 'es' | 'fr' | 'de' | 'pt' | 'zh'

export const translations: Record<SupportedLanguage, Record<string, string>> = {
  en: {
    // Right Rail Actions
    like: 'Like',
    comment: 'Comment',
    save: 'Save',
    play: 'Play',
    pause: 'Pause',
    next: 'Next',
    mute: 'Mute',
    unmute: 'Unmute',
    captions: 'Captions',
    settings: 'Settings',
    
    // Left Rail
    liveClass: 'Live Class',
    learningNow: 'learning now',
    shareLesson: 'Share this lesson',
    askKelly: 'Ask Kelly',
    searchLessons: 'Search 365 lessons...',
    todaysLesson: "Today's lesson with Kelly",
    copied: 'Copied!',
    copyLink: 'Copy Link',
    
    // Bottom Bar
    startLearning: 'Start Learning',
    continueLesson: 'Continue',
    isThisTrue: 'Is this',
    true: 'True',
    or: 'or',
    false: 'False',
    
    // Day Picker
    day: 'Day',
    of: 'of',
    
    // Controls
    age: 'Age',
    tone: 'Tone',
    language: 'Lang',
    
    // Modes
    live: 'Live',
    kelly: 'Kelly',
    find: 'Find',
    
    // Loading - Fun rotating messages (keep short to avoid overflow)
    loadingMessage: "Pardon while my systems come online...",
    loadingMessage1: "Gathering today's adventure...",
    loadingMessage2: "Connecting the dots...",
    loadingMessage3: "Loading something wonderful...",
    loadingMessage4: "Almost ready to learn together!",
    loadingMessage5: "Warming up my curiosity...",
    
    // Right Rail (i18n)
    profile: 'Profile',
    share: 'Share',
    menu: 'Menu',
    ziggurat: 'Ziggurat',
    
    // Phase labels
    phaseHook: 'Hook',
    phaseStory: 'Story',
    phaseWonder: 'Wonder',
    phaseAction: 'Try It',
    phaseWisdom: 'Wisdom',
    
    // Age labels
    ageKids: 'Kids',
    ageTeens: 'Teens',
    ageYoung: 'Young',
    ageAdult: 'Adult',
    ageSenior: 'Senior',
    
    // Coming Soon / Content Status
    comingSoon: 'Coming Soon',
    contentGenerating: 'Video generating...',
    audioOnly: 'Audio lesson',
    videoReady: 'Video ready',
    lessonDay: 'Day',
    
    // Date formatting
    today: 'Today',
    tomorrow: 'Tomorrow',
    yesterday: 'Yesterday',
  },
  es: {
    // Right Rail Actions
    like: 'Me gusta',
    comment: 'Comentar',
    save: 'Guardar',
    play: 'Reproducir',
    pause: 'Pausar',
    next: 'Siguiente',
    mute: 'Silenciar',
    unmute: 'Activar',
    captions: 'Subtítulos',
    settings: 'Ajustes',
    
    // Left Rail
    liveClass: 'Clase en Vivo',
    learningNow: 'aprendiendo ahora',
    shareLesson: 'Compartir lección',
    askKelly: 'Pregunta a Kelly',
    searchLessons: 'Buscar 365 lecciones...',
    todaysLesson: 'Lección de hoy con Kelly',
    copied: '¡Copiado!',
    copyLink: 'Copiar',
    
    // Bottom Bar
    startLearning: 'Empezar',
    continueLesson: 'Continuar',
    isThisTrue: '¿Es esto',
    true: 'Verdad',
    or: 'o',
    false: 'Falso',
    
    // Day Picker
    day: 'Día',
    of: 'de',
    
    // Controls
    age: 'Edad',
    tone: 'Tono',
    language: 'Idioma',
    
    // Modes
    live: 'Vivo',
    kelly: 'Kelly',
    find: 'Buscar',
    
    // Loading - Fun rotating messages
    loadingMessage: "Perdona mientras mis sistemas se activan...",
    loadingMessage1: "Preparando la aventura de hoy...",
    loadingMessage2: "Conectando los puntos...",
    loadingMessage3: "Cargando algo maravilloso...",
    loadingMessage4: "¡Casi listos para aprender juntos!",
    loadingMessage5: "Calentando mi curiosidad...",
    
    // Right Rail
    profile: 'Perfil',
    share: 'Compartir',
    menu: 'Menú',
    ziggurat: 'Zigurat',
    
    // Phase labels
    phaseHook: 'Chispa',
    phaseStory: 'Historia',
    phaseWonder: 'Asombro',
    phaseAction: 'Inténtalo',
    phaseWisdom: 'Sabiduría',
    
    // Age labels
    ageKids: 'Niños',
    ageTeens: 'Adolescentes',
    ageYoung: 'Jóvenes',
    ageAdult: 'Adultos',
    ageSenior: 'Mayores',
    
    // Coming Soon / Content Status
    comingSoon: 'Próximamente',
    contentGenerating: 'Generando video...',
    audioOnly: 'Lección de audio',
    videoReady: 'Video listo',
    lessonDay: 'Día',
    
    // Date formatting
    today: 'Hoy',
    tomorrow: 'Mañana',
    yesterday: 'Ayer',
  },
  fr: {
    // Right Rail Actions
    like: 'Aimer',
    comment: 'Commenter',
    save: 'Sauver',
    play: 'Lire',
    pause: 'Pause',
    next: 'Suivant',
    mute: 'Muet',
    unmute: 'Son',
    captions: 'Sous-titres',
    settings: 'Paramètres',
    
    // Left Rail
    liveClass: 'Classe en Direct',
    learningNow: 'apprennent',
    shareLesson: 'Partager la leçon',
    askKelly: 'Demandez à Kelly',
    searchLessons: 'Rechercher 365 leçons...',
    todaysLesson: 'La leçon du jour avec Kelly',
    copied: 'Copié!',
    copyLink: 'Copier',
    
    // Bottom Bar
    startLearning: 'Commencer',
    continueLesson: 'Continuer',
    isThisTrue: 'Est-ce',
    true: 'Vrai',
    or: 'ou',
    false: 'Faux',
    
    // Day Picker
    day: 'Jour',
    of: 'de',
    
    // Controls
    age: 'Âge',
    tone: 'Ton',
    language: 'Langue',
    
    // Modes
    live: 'Direct',
    kelly: 'Kelly',
    find: 'Chercher',
    
    // Coming Soon / Content Status
    comingSoon: 'Bientôt',
    contentGenerating: 'Vidéo en cours...',
    audioOnly: 'Leçon audio',
    videoReady: 'Vidéo prête',
    lessonDay: 'Jour',
    today: "Aujourd'hui",
    tomorrow: 'Demain',
    yesterday: 'Hier',
  },
  de: {
    // Right Rail Actions
    like: 'Gefällt mir',
    comment: 'Kommentar',
    save: 'Speichern',
    play: 'Abspielen',
    pause: 'Pause',
    next: 'Weiter',
    mute: 'Stumm',
    unmute: 'Ton an',
    captions: 'Untertitel',
    settings: 'Einstellungen',
    
    // Left Rail
    liveClass: 'Live-Kurs',
    learningNow: 'lernen gerade',
    shareLesson: 'Lektion teilen',
    askKelly: 'Frag Kelly',
    searchLessons: '365 Lektionen suchen...',
    
    // Bottom Bar
    startLearning: 'Starten',
    continueLesson: 'Fortsetzen',
    isThisTrue: 'Ist das',
    true: 'Wahr',
    or: 'oder',
    false: 'Falsch',
    
    // Day Picker
    day: 'Tag',
    of: 'von',
    
    // Controls
    age: 'Alter',
    tone: 'Ton',
    language: 'Sprache',
    
    // Modes
    live: 'Live',
    kelly: 'Kelly',
    find: 'Suchen',
    
    // Coming Soon / Content Status
    comingSoon: 'Demnächst',
    contentGenerating: 'Video wird erstellt...',
    audioOnly: 'Audio-Lektion',
    videoReady: 'Video bereit',
    lessonDay: 'Tag',
    today: 'Heute',
    tomorrow: 'Morgen',
    yesterday: 'Gestern',
  },
  pt: {
    // Right Rail Actions
    like: 'Curtir',
    comment: 'Comentar',
    save: 'Salvar',
    play: 'Reproduzir',
    pause: 'Pausar',
    next: 'Próximo',
    mute: 'Mudo',
    unmute: 'Som',
    captions: 'Legendas',
    settings: 'Configurações',
    
    // Left Rail
    liveClass: 'Aula ao Vivo',
    learningNow: 'aprendendo agora',
    shareLesson: 'Compartilhar lição',
    askKelly: 'Pergunte à Kelly',
    searchLessons: 'Pesquisar 365 lições...',
    
    // Bottom Bar
    startLearning: 'Começar',
    continueLesson: 'Continuar',
    isThisTrue: 'Isso é',
    true: 'Verdade',
    or: 'ou',
    false: 'Falso',
    
    // Day Picker
    day: 'Dia',
    of: 'de',
    
    // Controls
    age: 'Idade',
    tone: 'Tom',
    language: 'Idioma',
    
    // Modes
    live: 'Ao Vivo',
    kelly: 'Kelly',
    find: 'Buscar',
    
    // Coming Soon / Content Status
    comingSoon: 'Em breve',
    contentGenerating: 'Gerando vídeo...',
    audioOnly: 'Aula em áudio',
    videoReady: 'Vídeo pronto',
    lessonDay: 'Dia',
    today: 'Hoje',
    tomorrow: 'Amanhã',
    yesterday: 'Ontem',
  },
  zh: {
    // Right Rail Actions
    like: '喜欢',
    comment: '评论',
    save: '收藏',
    play: '播放',
    pause: '暂停',
    next: '下一个',
    mute: '静音',
    unmute: '取消静音',
    captions: '字幕',
    settings: '设置',
    
    // Left Rail
    liveClass: '直播课程',
    learningNow: '正在学习',
    shareLesson: '分享课程',
    askKelly: '问 Kelly',
    searchLessons: '搜索365节课...',
    
    // Bottom Bar
    startLearning: '开始学习',
    continueLesson: '继续',
    isThisTrue: '这是',
    true: '真',
    or: '还是',
    false: '假',
    
    // Day Picker
    day: '第',
    of: '天，共',
    
    // Controls
    age: '年龄',
    tone: '语气',
    language: '语言',
    
    // Modes
    live: '直播',
    kelly: 'Kelly',
    find: '搜索',
    
    // Coming Soon / Content Status
    comingSoon: '即将推出',
    contentGenerating: '视频生成中...',
    audioOnly: '音频课程',
    videoReady: '视频已就绪',
    lessonDay: '第',
    today: '今天',
    tomorrow: '明天',
    yesterday: '昨天',
  },
}

/**
 * Get a translation for a key in the specified language
 */
export function t(key: string, lang: string): string {
  const language = (lang as SupportedLanguage) || 'en'
  return translations[language]?.[key] || translations.en[key] || key
}

/**
 * Create a translation function bound to a specific language
 */
export function useTranslation(lang: string) {
  return (key: string) => t(key, lang)
}
