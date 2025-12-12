import { enUS } from './en-us';
import { esES } from './es-es';
import { ptBR } from './pt-br';
import type { LocaleCode, LocaleDictionary } from './types';

export const defaultLocale: LocaleCode = 'en-US';

const localeMap: Record<LocaleCode, LocaleDictionary> = {
  'en-US': enUS,
  'es-ES': esES,
  'pt-BR': ptBR
};

const localePrefixes: Record<LocaleCode, string> = {
  'en-US': '',
  'es-ES': 'es-es',
  'pt-BR': 'pt-br'
};

export const supportedLocales: { code: LocaleCode; label: string; prefix: string }[] = [
  { code: 'en-US', label: 'English', prefix: '' },
  { code: 'es-ES', label: 'Español', prefix: 'es-es' },
  { code: 'pt-BR', label: 'Português', prefix: 'pt-br' }
];

export function getDictionary(locale: string | null | undefined): LocaleDictionary {
  if (!locale) {
    return localeMap[defaultLocale];
  }
  const normalized = locale as LocaleCode;
  return localeMap[normalized] ?? localeMap[defaultLocale];
}

export function normaliseLocale(locale: string | null | undefined): LocaleCode {
  if (!locale) {
    return defaultLocale;
  }
  const normalised = locale.toLowerCase();
  const match = supportedLocales.find(({ code, prefix }) => {
    return code.toLowerCase() === normalised || prefix === normalised;
  });
  return match?.code ?? defaultLocale;
}

export function getLocaleFromPath(pathname: string): LocaleCode {
  if (!pathname) return defaultLocale;
  const trimmed = pathname.replace(/^\//, '');
  const [first] = trimmed.split('/');
  const match = supportedLocales.find((locale) => locale.prefix === first);
  return match?.code ?? defaultLocale;
}

export function stripLocaleFromPath(pathname: string): string {
  if (!pathname) return '';
  const locale = getLocaleFromPath(pathname);
  const prefix = localePrefixes[locale];
  if (!prefix) {
    return pathname.replace(/^\/+/, '');
  }

  const pattern = new RegExp(`^/${prefix}/?`);
  return pathname.replace(pattern, '');
}

export function buildLocalizedPath(pathname: string, targetLocale: LocaleCode): string {
  if (!pathname) return '/';
  const base = stripLocaleFromPath(pathname);
  const normalized = base.replace(/\/+$/, '');
  const safe = normalized === '' ? '' : `${normalized}/`;
  const prefix = localePrefixes[targetLocale];
  if (!prefix) {
    return `/${safe}`;
  }
  return `/${prefix}/${safe}`.replace(/\/+/g, '/');
}

export function getAlternateHreflang(pathname: string): { locale: LocaleCode; href: string }[] {
  return supportedLocales.map((locale) => ({
    locale: locale.code,
    href: new URL(buildLocalizedPath(pathname, locale.code), 'https://www.thedailylesson.com').toString()
  }));
}

