import type { VercelRequest, VercelResponse } from '@vercel/node';

/**
 * Geo-Context API
 * GET /api/geo-context
 * 
 * Returns comprehensive context about the user's location, time, and season.
 * Kelly uses this to personalize greetings and lesson presentations.
 * 
 * Returns:
 * - Country and region
 * - Local time and timezone
 * - Time of day (morning/afternoon/evening/night)
 * - Day of week and weekend detection
 * - Season (hemisphere-aware!)
 * - Day of year (for lesson alignment)
 */

interface GeoContext {
  // Location
  country: string;
  countryName: string;
  region: string;
  city: string;
  
  // Time
  timezone: string;
  localTime: string;
  localDate: string;
  hour: number;
  minute: number;
  timeOfDay: 'morning' | 'afternoon' | 'evening' | 'night';
  
  // Calendar
  dayOfWeek: string;
  dayOfWeekShort: string;
  dayOfWeekNumber: number; // 0 = Sunday
  dayOfMonth: number;
  dayOfYear: number;
  weekOfYear: number;
  month: string;
  monthNumber: number; // 0-11
  year: number;
  isWeekend: boolean;
  
  // Season (hemisphere-aware!)
  season: 'spring' | 'summer' | 'autumn' | 'winter';
  hemisphere: 'northern' | 'southern';
  
  // Language hints
  suggestedLanguage: string;
  
  // Formatting preferences
  dateFormat: 'MM/DD/YYYY' | 'DD/MM/YYYY' | 'YYYY-MM-DD';
  use24Hour: boolean;
}

// Countries in the Southern Hemisphere (for season calculation)
const SOUTHERN_HEMISPHERE = new Set([
  'AU', 'NZ', 'AR', 'CL', 'UY', 'PY', 'BO', 'PE', 'ZA', 
  'NA', 'BW', 'ZW', 'MZ', 'MG', 'ID', 'TL', 'PG'
]);

// Partial Southern Hemisphere (equatorial or mostly southern)
const PARTIAL_SOUTHERN = new Set(['BR', 'EC', 'CO', 'KE', 'TZ', 'UG']);

// Countries that use DD/MM/YYYY date format
const DDMMYYYY_COUNTRIES = new Set([
  'GB', 'AU', 'NZ', 'IE', 'IN', 'ZA', 'DE', 'FR', 'IT', 'ES', 
  'PT', 'NL', 'BE', 'AT', 'CH', 'PL', 'RU', 'BR', 'AR', 'MX'
]);

// Countries that primarily use 24-hour time
const USE_24_HOUR = new Set([
  'DE', 'FR', 'IT', 'ES', 'PT', 'NL', 'BE', 'AT', 'CH', 'PL', 
  'RU', 'JP', 'KR', 'CN', 'SE', 'NO', 'DK', 'FI', 'BR', 'AR'
]);

// Country to suggested language mapping
const COUNTRY_TO_LANGUAGE: Record<string, string> = {
  US: 'en', GB: 'en', CA: 'en', AU: 'en', NZ: 'en', IE: 'en',
  ES: 'es', MX: 'es', AR: 'es', CO: 'es', PE: 'es', CL: 'es', VE: 'es', EC: 'es',
  BR: 'pt', PT: 'pt',
  FR: 'fr', BE: 'fr', CH: 'fr', CA: 'fr', // Quebec
  DE: 'de', AT: 'de', CH: 'de',
  IN: 'hi', // Though English is common too
};

// Country code to name mapping (common ones)
const COUNTRY_NAMES: Record<string, string> = {
  US: 'United States', GB: 'United Kingdom', CA: 'Canada', AU: 'Australia',
  NZ: 'New Zealand', IE: 'Ireland', DE: 'Germany', FR: 'France', ES: 'Spain',
  IT: 'Italy', NL: 'Netherlands', BE: 'Belgium', AT: 'Austria', CH: 'Switzerland',
  PT: 'Portugal', PL: 'Poland', SE: 'Sweden', NO: 'Norway', DK: 'Denmark',
  FI: 'Finland', RU: 'Russia', JP: 'Japan', KR: 'South Korea', CN: 'China',
  IN: 'India', BR: 'Brazil', MX: 'Mexico', AR: 'Argentina', CL: 'Chile',
  CO: 'Colombia', PE: 'Peru', ZA: 'South Africa', EG: 'Egypt', NG: 'Nigeria',
  KE: 'Kenya', TH: 'Thailand', VN: 'Vietnam', PH: 'Philippines', ID: 'Indonesia',
  MY: 'Malaysia', SG: 'Singapore', HK: 'Hong Kong', TW: 'Taiwan',
};

function getCountryName(code: string): string {
  return COUNTRY_NAMES[code] || code;
}

function getSeason(month: number, hemisphere: 'northern' | 'southern'): 'spring' | 'summer' | 'autumn' | 'winter' {
  // Northern Hemisphere seasons
  const northernSeasons: Record<number, 'spring' | 'summer' | 'autumn' | 'winter'> = {
    0: 'winter', 1: 'winter', 2: 'spring',
    3: 'spring', 4: 'spring', 5: 'summer',
    6: 'summer', 7: 'summer', 8: 'autumn',
    9: 'autumn', 10: 'autumn', 11: 'winter'
  };
  
  const season = northernSeasons[month];
  
  // Flip for Southern Hemisphere
  if (hemisphere === 'southern') {
    const flip: Record<string, 'spring' | 'summer' | 'autumn' | 'winter'> = {
      spring: 'autumn',
      summer: 'winter',
      autumn: 'spring',
      winter: 'summer'
    };
    return flip[season];
  }
  
  return season;
}

function getTimeOfDay(hour: number): 'morning' | 'afternoon' | 'evening' | 'night' {
  if (hour >= 5 && hour < 12) return 'morning';
  if (hour >= 12 && hour < 17) return 'afternoon';
  if (hour >= 17 && hour < 21) return 'evening';
  return 'night';
}

function getDayOfYear(date: Date): number {
  const start = new Date(date.getFullYear(), 0, 0);
  const diff = date.getTime() - start.getTime();
  const oneDay = 1000 * 60 * 60 * 24;
  return Math.floor(diff / oneDay);
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Cache-Control', 'public, max-age=300'); // 5 min cache
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  try {
    // Get geo data from Vercel headers
    const country = (req.headers['x-vercel-ip-country'] as string) || 'US';
    const region = (req.headers['x-vercel-ip-country-region'] as string) || '';
    const city = decodeURIComponent((req.headers['x-vercel-ip-city'] as string) || '');
    const timezone = (req.headers['x-vercel-ip-timezone'] as string) || 'America/New_York';
    
    // Allow override for testing
    const forceCountry = req.query.force_country as string;
    const forceTimezone = req.query.force_timezone as string;
    const effectiveCountry = forceCountry?.toUpperCase() || country;
    const effectiveTimezone = forceTimezone || timezone;
    
    // Get current time in user's timezone
    const now = new Date();
    const options: Intl.DateTimeFormatOptions = { timeZone: effectiveTimezone };
    
    // Parse local time components
    const localTimeStr = now.toLocaleString('en-US', { 
      ...options, 
      hour12: false,
      hour: '2-digit',
      minute: '2-digit'
    });
    const [hourStr, minuteStr] = localTimeStr.split(':');
    const hour = parseInt(hourStr, 10);
    const minute = parseInt(minuteStr, 10);
    
    // Get day of week (getDay() returns 0=Sunday, 1=Monday, etc.)
    // Use the Date object directly for day number since 'weekday: numeric' is not valid
    const formatter = new Intl.DateTimeFormat('en-US', { ...options });
    const parts = formatter.formatToParts(now);
    const dayOfWeek = now.toLocaleString('en-US', { ...options, weekday: 'long' });
    const dayOfWeekShort = now.toLocaleString('en-US', { ...options, weekday: 'short' });
    // Calculate day of week from the formatted date
    const dayOfWeekNumber = new Date(now.toLocaleString('en-US', { ...options })).getDay();
    
    // Get date components
    const dayOfMonth = parseInt(now.toLocaleString('en-US', { ...options, day: 'numeric' }), 10);
    const monthNumber = parseInt(now.toLocaleString('en-US', { ...options, month: 'numeric' }), 10) - 1;
    const month = now.toLocaleString('en-US', { ...options, month: 'long' });
    const year = parseInt(now.toLocaleString('en-US', { ...options, year: 'numeric' }), 10);
    
    // Determine hemisphere
    let hemisphere: 'northern' | 'southern' = 'northern';
    if (SOUTHERN_HEMISPHERE.has(effectiveCountry)) {
      hemisphere = 'southern';
    } else if (PARTIAL_SOUTHERN.has(effectiveCountry)) {
      // For equatorial countries, could be more nuanced
      // For now, treat as northern
      hemisphere = 'northern';
    }
    
    // Build context
    const context: GeoContext = {
      // Location
      country: effectiveCountry,
      countryName: getCountryName(effectiveCountry),
      region,
      city,
      
      // Time
      timezone: effectiveTimezone,
      localTime: now.toLocaleString('en-US', { ...options, timeStyle: 'short' }),
      localDate: now.toLocaleString('en-US', { ...options, dateStyle: 'medium' }),
      hour,
      minute,
      timeOfDay: getTimeOfDay(hour),
      
      // Calendar
      dayOfWeek,
      dayOfWeekShort,
      dayOfWeekNumber,
      dayOfMonth,
      dayOfYear: getDayOfYear(now),
      weekOfYear: Math.ceil(getDayOfYear(now) / 7),
      month,
      monthNumber,
      year,
      isWeekend: dayOfWeekNumber === 0 || dayOfWeekNumber === 6,
      
      // Season
      season: getSeason(monthNumber, hemisphere),
      hemisphere,
      
      // Language hints
      suggestedLanguage: COUNTRY_TO_LANGUAGE[effectiveCountry] || 'en',
      
      // Formatting preferences
      dateFormat: DDMMYYYY_COUNTRIES.has(effectiveCountry) ? 'DD/MM/YYYY' : 'MM/DD/YYYY',
      use24Hour: USE_24_HOUR.has(effectiveCountry),
    };
    
    return res.status(200).json(context);
    
  } catch (error) {
    console.error('[geo-context] Error:', error);
    
    // Return safe defaults
    const now = new Date();
    return res.status(200).json({
      country: 'US',
      countryName: 'United States',
      region: '',
      city: '',
      timezone: 'America/New_York',
      localTime: now.toLocaleTimeString('en-US', { timeStyle: 'short' }),
      localDate: now.toLocaleDateString('en-US', { dateStyle: 'medium' }),
      hour: now.getHours(),
      minute: now.getMinutes(),
      timeOfDay: getTimeOfDay(now.getHours()),
      dayOfWeek: now.toLocaleDateString('en-US', { weekday: 'long' }),
      dayOfWeekShort: now.toLocaleDateString('en-US', { weekday: 'short' }),
      dayOfWeekNumber: now.getDay(),
      dayOfMonth: now.getDate(),
      dayOfYear: getDayOfYear(now),
      weekOfYear: Math.ceil(getDayOfYear(now) / 7),
      month: now.toLocaleDateString('en-US', { month: 'long' }),
      monthNumber: now.getMonth(),
      year: now.getFullYear(),
      isWeekend: now.getDay() === 0 || now.getDay() === 6,
      season: getSeason(now.getMonth(), 'northern'),
      hemisphere: 'northern',
      suggestedLanguage: 'en',
      dateFormat: 'MM/DD/YYYY',
      use24Hour: false,
    } as GeoContext);
  }
}
