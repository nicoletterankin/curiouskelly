/**
 * 🌐 i18n Strings for Ziggurat Vision Component
 * 
 * Add these to your main i18n config or use directly.
 * Follows same pattern as v0's i18n additions.
 */

export const ZIGGURAT_I18N = {
  en: {
    'ziggurat.title': 'Ziggurat LED Vision',
    'ziggurat.before': 'Before',
    'ziggurat.after': 'After',
    'ziggurat.palette': 'Palette',
    'ziggurat.time': 'Time of Day',
    'ziggurat.download': 'Download',
    'ziggurat.download.1080p': 'Download 1080p',
    'ziggurat.download.4k': 'Download 4K',
    'ziggurat.download.full': 'Download Full Resolution',
    'ziggurat.palette.rainbow': 'Rainbow',
    'ziggurat.palette.cool': 'Cool Tones',
    'ziggurat.palette.warm': 'Warm Tones',
    'ziggurat.palette.white': 'Warm White',
    'ziggurat.palette.gold': 'Gold',
    'ziggurat.palette.cyan': 'Cyan',
    'ziggurat.palette.usa': 'USA',
    'ziggurat.time.night': 'Night',
    'ziggurat.time.late-night': 'Late Night',
    'ziggurat.time.twilight': 'Twilight',
    'ziggurat.time.dusk': 'Dusk',
    'ziggurat.tiers': '7 Precision-Traced Tiers',
    'ziggurat.building': 'Chet Holifield Federal Building',
  },
  es: {
    'ziggurat.title': 'Visión LED del Zigurat',
    'ziggurat.before': 'Antes',
    'ziggurat.after': 'Después',
    'ziggurat.palette': 'Paleta',
    'ziggurat.time': 'Hora del Día',
    'ziggurat.download': 'Descargar',
    'ziggurat.download.1080p': 'Descargar 1080p',
    'ziggurat.download.4k': 'Descargar 4K',
    'ziggurat.download.full': 'Descargar Resolución Completa',
    'ziggurat.palette.rainbow': 'Arcoíris',
    'ziggurat.palette.cool': 'Tonos Fríos',
    'ziggurat.palette.warm': 'Tonos Cálidos',
    'ziggurat.palette.white': 'Blanco Cálido',
    'ziggurat.palette.gold': 'Dorado',
    'ziggurat.palette.cyan': 'Cian',
    'ziggurat.palette.usa': 'EE.UU.',
    'ziggurat.time.night': 'Noche',
    'ziggurat.time.late-night': 'Noche Tardía',
    'ziggurat.time.twilight': 'Crepúsculo',
    'ziggurat.time.dusk': 'Atardecer',
    'ziggurat.tiers': '7 Niveles Trazados con Precisión',
    'ziggurat.building': 'Edificio Federal Chet Holifield',
  },
  fr: {
    'ziggurat.title': 'Vision LED de la Ziggourat',
    'ziggurat.before': 'Avant',
    'ziggurat.after': 'Après',
    'ziggurat.palette': 'Palette',
    'ziggurat.time': 'Moment de la Journée',
    'ziggurat.download': 'Télécharger',
    'ziggurat.download.1080p': 'Télécharger 1080p',
    'ziggurat.download.4k': 'Télécharger 4K',
    'ziggurat.download.full': 'Télécharger Pleine Résolution',
    'ziggurat.palette.rainbow': 'Arc-en-ciel',
    'ziggurat.palette.cool': 'Tons Froids',
    'ziggurat.palette.warm': 'Tons Chauds',
    'ziggurat.palette.white': 'Blanc Chaud',
    'ziggurat.palette.gold': 'Or',
    'ziggurat.palette.cyan': 'Cyan',
    'ziggurat.palette.usa': 'USA',
    'ziggurat.time.night': 'Nuit',
    'ziggurat.time.late-night': 'Nuit Tardive',
    'ziggurat.time.twilight': 'Crépuscule',
    'ziggurat.time.dusk': 'Aube',
    'ziggurat.tiers': '7 Niveaux Tracés avec Précision',
    'ziggurat.building': 'Bâtiment Fédéral Chet Holifield',
  },
  zh: {
    'ziggurat.title': '金字塔LED愿景',
    'ziggurat.before': '之前',
    'ziggurat.after': '之后',
    'ziggurat.palette': '调色板',
    'ziggurat.time': '时间',
    'ziggurat.download': '下载',
    'ziggurat.download.1080p': '下载 1080p',
    'ziggurat.download.4k': '下载 4K',
    'ziggurat.download.full': '下载完整分辨率',
    'ziggurat.palette.rainbow': '彩虹',
    'ziggurat.palette.cool': '冷色调',
    'ziggurat.palette.warm': '暖色调',
    'ziggurat.palette.white': '暖白',
    'ziggurat.palette.gold': '金色',
    'ziggurat.palette.cyan': '青色',
    'ziggurat.palette.usa': '美国',
    'ziggurat.time.night': '夜晚',
    'ziggurat.time.late-night': '深夜',
    'ziggurat.time.twilight': '黄昏',
    'ziggurat.time.dusk': '傍晚',
    'ziggurat.tiers': '7层精确追踪',
    'ziggurat.building': '切特·霍利菲尔德联邦大楼',
  },
};

export type ZigguratI18nKey = keyof typeof ZIGGURAT_I18N.en;
export type SupportedLanguage = keyof typeof ZIGGURAT_I18N;

/**
 * Get translated string
 */
export function t(key: ZigguratI18nKey, lang: SupportedLanguage = 'en'): string {
  return ZIGGURAT_I18N[lang]?.[key] || ZIGGURAT_I18N.en[key] || key;
}
