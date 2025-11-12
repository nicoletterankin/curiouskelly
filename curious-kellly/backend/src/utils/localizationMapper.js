/**
 * Localization Mapper
 * Builds locale-specific views from Lesson/PhaseDNA objects that use *_translations keys.
 *
 * Downstream engines expect to request localized copies of the DNA without
 * manually traversing every translation map. This utility produces those
 * fully resolved views while preserving the original authoring format.
 */

const DEFAULT_FALLBACK_LANGUAGE = 'en';
const DEFAULT_SUPPORTED_LANGUAGES = ['en', 'es', 'fr'];

/**
 * Collect every language code referenced inside *_translations maps.
 * @param {unknown} node
 * @param {Set<string>} accumulator
 * @returns {Set<string>}
 */
function collectSupportedLanguages(node, accumulator = new Set()) {
  if (!node || typeof node !== 'object') {
    return accumulator;
  }

  if (Array.isArray(node)) {
    node.forEach(item => collectSupportedLanguages(item, accumulator));
    return accumulator;
  }

  for (const [key, value] of Object.entries(node)) {
    if (key.endsWith('_translations') && value && typeof value === 'object' && !Array.isArray(value)) {
      Object.keys(value).forEach(lang => accumulator.add(lang));
    } else {
      collectSupportedLanguages(value, accumulator);
    }
  }

  return accumulator;
}

/**
 * Resolve a translation for the requested locale with graceful fallback.
 * @param {Record<string, unknown>} translations
 * @param {string} locale
 * @returns {unknown | undefined}
 */
function resolveTranslation(translations, locale) {
  if (!translations || typeof translations !== 'object') {
    return undefined;
  }

  if (Object.prototype.hasOwnProperty.call(translations, locale)) {
    return translations[locale];
  }

  if (locale !== DEFAULT_FALLBACK_LANGUAGE && Object.prototype.hasOwnProperty.call(translations, DEFAULT_FALLBACK_LANGUAGE)) {
    return translations[DEFAULT_FALLBACK_LANGUAGE];
  }

  const availableValues = Object.values(translations);
  return availableValues.find(value => value !== undefined && value !== null);
}

/**
 * Produce a deep localized clone of the provided node.
 * @param {unknown} node
 * @param {string} locale
 * @returns {unknown}
 */
function applyLocale(node, locale) {
  if (Array.isArray(node)) {
    return node.map(item => applyLocale(item, locale));
  }

  if (!node || typeof node !== 'object') {
    return node;
  }

  const result = Array.isArray(node) ? [] : {};

  for (const [key, value] of Object.entries(node)) {
    if (key.endsWith('_translations')) {
      // Skip the translation map itself; handled alongside its base key.
      continue;
    }

    const translationsKey = `${key}_translations`;
    const translations = node[translationsKey];

    let resolvedValue = value;
    if (translations && typeof translations === 'object' && !Array.isArray(translations)) {
      const localized = resolveTranslation(translations, locale);
      if (localized !== undefined) {
        resolvedValue = localized;
      }
    }

    result[key] = applyLocale(resolvedValue, locale);
  }

  return result;
}

/**
 * Build localized views and metadata for a given lesson/DNA object.
 * @param {object} lesson
 * @param {string[]} preferredLanguages
 * @returns {{ supportedLanguages: string[], locales: Record<string, unknown> }}
 */
function buildLocalizationBundle(lesson, preferredLanguages = DEFAULT_SUPPORTED_LANGUAGES) {
  const detectedLanguages = collectSupportedLanguages(lesson);
  preferredLanguages.forEach(lang => detectedLanguages.add(lang));

  // Ensure fallback language is always present
  detectedLanguages.add(DEFAULT_FALLBACK_LANGUAGE);

  const supportedLanguages = Array.from(detectedLanguages);
  const locales = supportedLanguages.reduce((accumulator, locale) => {
    accumulator[locale] = applyLocale(lesson, locale);
    return accumulator;
  }, {});

  return {
    supportedLanguages,
    locales
  };
}

module.exports = {
  buildLocalizationBundle,
  applyLocale,
  collectSupportedLanguages,
  DEFAULT_FALLBACK_LANGUAGE
};

