/**
 * INTERNATIONAL PRICING CONFIGURATION
 * Single source of truth for all pricing, currencies, and payment methods.
 * 
 * To add a new region:
 * 1. Add to COUNTRY_TO_CURRENCY
 * 2. Add price IDs to PRICE_IDS (after creating in Stripe)
 * 3. Add display prices to DISPLAY_PRICES
 * 4. Add payment methods to PAYMENT_METHODS if applicable
 */

// ============================================
// COUNTRY → CURRENCY MAPPING
// ============================================

export const COUNTRY_TO_CURRENCY: Record<string, string> = {
  // North America
  US: 'USD',
  CA: 'CAD',
  
  // UK & Ireland
  GB: 'GBP',
  IE: 'EUR', // Ireland uses Euro
  
  // Eurozone
  AT: 'EUR', // Austria
  BE: 'EUR', // Belgium
  CY: 'EUR', // Cyprus
  EE: 'EUR', // Estonia
  FI: 'EUR', // Finland
  FR: 'EUR', // France
  DE: 'EUR', // Germany
  GR: 'EUR', // Greece
  IT: 'EUR', // Italy
  LV: 'EUR', // Latvia
  LT: 'EUR', // Lithuania
  LU: 'EUR', // Luxembourg
  MT: 'EUR', // Malta
  NL: 'EUR', // Netherlands
  PT: 'EUR', // Portugal
  SK: 'EUR', // Slovakia
  SI: 'EUR', // Slovenia
  ES: 'EUR', // Spain
  
  // Other developed markets
  AU: 'AUD', // Australia
  NZ: 'NZD', // New Zealand (use AUD pricing)
  CH: 'CHF', // Switzerland
  
  // PPP Markets (discounted)
  IN: 'INR', // India - 50% PPP
  BR: 'BRL', // Brazil - 40% PPP
  MX: 'MXN', // Mexico - 35% PPP
  PL: 'PLN', // Poland - 30% PPP
  TR: 'TRY', // Turkey - 45% PPP
  ID: 'IDR', // Indonesia - 55% PPP
  PH: 'PHP', // Philippines - 45% PPP
  ZA: 'ZAR', // South Africa - 40% PPP
  AR: 'ARS', // Argentina - 60% PPP
  CO: 'COP', // Colombia - 40% PPP
  CL: 'CLP', // Chile - 30% PPP
  PK: 'PKR', // Pakistan - 55% PPP
  VN: 'VND', // Vietnam - 50% PPP
  TH: 'THB', // Thailand - 35% PPP
  MY: 'MYR', // Malaysia - 30% PPP
  
  // East Asia (parity or slight premium)
  JP: 'JPY', // Japan
  KR: 'KRW', // South Korea
  SG: 'SGD', // Singapore
  HK: 'HKD', // Hong Kong
  TW: 'TWD', // Taiwan
};

// Countries eligible for PPP (Purchasing Power Parity) discounts
export const PPP_COUNTRIES = new Set([
  'IN', 'BR', 'MX', 'PL', 'TR', 'ID', 'PH', 'ZA', 
  'AR', 'CO', 'CL', 'PK', 'VN', 'TH', 'MY'
]);

// EU countries for SEPA support
export const EU_COUNTRIES = new Set([
  'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
  'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL',
  'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'
]);

// ============================================
// STRIPE PRICE IDS BY CURRENCY
// TODO: Fill in after creating prices in Stripe Dashboard
// ============================================

export const PRICE_IDS: Record<string, Record<string, string>> = {
  USD: {
    monthly: process.env.STRIPE_PRICE_MONTHLY || '',
    annual: process.env.STRIPE_PRICE_ANNUAL || '',
    family: process.env.STRIPE_PRICE_FAMILY || '',
    lifetime: process.env.STRIPE_PRICE_LIFETIME || '',
    gift_3mo: process.env.STRIPE_PRICE_GIFT_3MO || '',
    gift_6mo: process.env.STRIPE_PRICE_GIFT_6MO || '',
    gift_12mo: process.env.STRIPE_PRICE_GIFT_12MO || '',
    gift_lifetime: process.env.STRIPE_PRICE_GIFT_LIFETIME || '',
  },
  EUR: {
    monthly: process.env.STRIPE_PRICE_MONTHLY_EUR || '',
    annual: process.env.STRIPE_PRICE_ANNUAL_EUR || '',
    family: process.env.STRIPE_PRICE_FAMILY_EUR || '',
    lifetime: process.env.STRIPE_PRICE_LIFETIME_EUR || '',
    gift_3mo: process.env.STRIPE_PRICE_GIFT_3MO_EUR || '',
    gift_6mo: process.env.STRIPE_PRICE_GIFT_6MO_EUR || '',
    gift_12mo: process.env.STRIPE_PRICE_GIFT_12MO_EUR || '',
    gift_lifetime: process.env.STRIPE_PRICE_GIFT_LIFETIME_EUR || '',
  },
  GBP: {
    monthly: process.env.STRIPE_PRICE_MONTHLY_GBP || '',
    annual: process.env.STRIPE_PRICE_ANNUAL_GBP || '',
    family: process.env.STRIPE_PRICE_FAMILY_GBP || '',
    lifetime: process.env.STRIPE_PRICE_LIFETIME_GBP || '',
    gift_3mo: process.env.STRIPE_PRICE_GIFT_3MO_GBP || '',
    gift_6mo: process.env.STRIPE_PRICE_GIFT_6MO_GBP || '',
    gift_12mo: process.env.STRIPE_PRICE_GIFT_12MO_GBP || '',
    gift_lifetime: process.env.STRIPE_PRICE_GIFT_LIFETIME_GBP || '',
  },
  CAD: {
    monthly: process.env.STRIPE_PRICE_MONTHLY_CAD || '',
    annual: process.env.STRIPE_PRICE_ANNUAL_CAD || '',
    family: process.env.STRIPE_PRICE_FAMILY_CAD || '',
    lifetime: process.env.STRIPE_PRICE_LIFETIME_CAD || '',
    gift_3mo: process.env.STRIPE_PRICE_GIFT_3MO_CAD || '',
    gift_6mo: process.env.STRIPE_PRICE_GIFT_6MO_CAD || '',
    gift_12mo: process.env.STRIPE_PRICE_GIFT_12MO_CAD || '',
    gift_lifetime: process.env.STRIPE_PRICE_GIFT_LIFETIME_CAD || '',
  },
  AUD: {
    monthly: process.env.STRIPE_PRICE_MONTHLY_AUD || '',
    annual: process.env.STRIPE_PRICE_ANNUAL_AUD || '',
    family: process.env.STRIPE_PRICE_FAMILY_AUD || '',
    lifetime: process.env.STRIPE_PRICE_LIFETIME_AUD || '',
    gift_3mo: process.env.STRIPE_PRICE_GIFT_3MO_AUD || '',
    gift_6mo: process.env.STRIPE_PRICE_GIFT_6MO_AUD || '',
    gift_12mo: process.env.STRIPE_PRICE_GIFT_12MO_AUD || '',
    gift_lifetime: process.env.STRIPE_PRICE_GIFT_LIFETIME_AUD || '',
  },
  // PPP Markets - Annual only initially (highest conversion)
  INR: {
    monthly: process.env.STRIPE_PRICE_MONTHLY_INR || '',
    annual: process.env.STRIPE_PRICE_ANNUAL_INR || '',
    lifetime: process.env.STRIPE_PRICE_LIFETIME_INR || '',
  },
  BRL: {
    monthly: process.env.STRIPE_PRICE_MONTHLY_BRL || '',
    annual: process.env.STRIPE_PRICE_ANNUAL_BRL || '',
    lifetime: process.env.STRIPE_PRICE_LIFETIME_BRL || '',
  },
  MXN: {
    monthly: process.env.STRIPE_PRICE_MONTHLY_MXN || '',
    annual: process.env.STRIPE_PRICE_ANNUAL_MXN || '',
    lifetime: process.env.STRIPE_PRICE_LIFETIME_MXN || '',
  },
  PLN: {
    monthly: process.env.STRIPE_PRICE_MONTHLY_PLN || '',
    annual: process.env.STRIPE_PRICE_ANNUAL_PLN || '',
    lifetime: process.env.STRIPE_PRICE_LIFETIME_PLN || '',
  },
};

// ============================================
// DISPLAY PRICES FOR UI
// ============================================

export const DISPLAY_PRICES: Record<string, {
  symbol: string;
  monthly: string;
  annual: string;
  family: string;
  lifetime: string;
  gift_3mo: string;
  gift_6mo: string;
  gift_12mo: string;
  gift_lifetime: string;
  perDay: string;
  savings: string;
}> = {
  USD: {
    symbol: '$',
    monthly: '$7.99',
    annual: '$49.99',
    family: '$99.99',
    lifetime: '$199.99',
    gift_3mo: '$24.99',
    gift_6mo: '$39.99',
    gift_12mo: '$49.99',
    gift_lifetime: '$149.99',
    perDay: '$0.14',
    savings: '48%',
  },
  EUR: {
    symbol: '€',
    monthly: '€7.99',
    annual: '€49.99',
    family: '€99.99',
    lifetime: '€199.99',
    gift_3mo: '€24.99',
    gift_6mo: '€39.99',
    gift_12mo: '€49.99',
    gift_lifetime: '€149.99',
    perDay: '€0.14',
    savings: '48%',
  },
  GBP: {
    symbol: '£',
    monthly: '£6.99',
    annual: '£44.99',
    family: '£89.99',
    lifetime: '£179.99',
    gift_3mo: '£21.99',
    gift_6mo: '£34.99',
    gift_12mo: '£44.99',
    gift_lifetime: '£129.99',
    perDay: '£0.12',
    savings: '48%',
  },
  CAD: {
    symbol: 'C$',
    monthly: 'C$10.99',
    annual: 'C$69.99',
    family: 'C$139.99',
    lifetime: 'C$279.99',
    gift_3mo: 'C$34.99',
    gift_6mo: 'C$54.99',
    gift_12mo: 'C$69.99',
    gift_lifetime: 'C$209.99',
    perDay: 'C$0.19',
    savings: '48%',
  },
  AUD: {
    symbol: 'A$',
    monthly: 'A$12.99',
    annual: 'A$79.99',
    family: 'A$159.99',
    lifetime: 'A$319.99',
    gift_3mo: 'A$39.99',
    gift_6mo: 'A$64.99',
    gift_12mo: 'A$79.99',
    gift_lifetime: 'A$239.99',
    perDay: 'A$0.22',
    savings: '48%',
  },
  // PPP Markets
  INR: {
    symbol: '₹',
    monthly: '₹299',
    annual: '₹1,999',
    family: '₹3,999',
    lifetime: '₹7,999',
    gift_3mo: '₹999',
    gift_6mo: '₹1,499',
    gift_12mo: '₹1,999',
    gift_lifetime: '₹5,999',
    perDay: '₹5.48',
    savings: '50%',
  },
  BRL: {
    symbol: 'R$',
    monthly: 'R$29.99',
    annual: 'R$199.99',
    family: 'R$399.99',
    lifetime: 'R$799.99',
    gift_3mo: 'R$99.99',
    gift_6mo: 'R$159.99',
    gift_12mo: 'R$199.99',
    gift_lifetime: 'R$599.99',
    perDay: 'R$0.55',
    savings: '40%',
  },
  MXN: {
    symbol: 'MX$',
    monthly: 'MX$99',
    annual: 'MX$699',
    family: 'MX$1,399',
    lifetime: 'MX$2,999',
    gift_3mo: 'MX$349',
    gift_6mo: 'MX$549',
    gift_12mo: 'MX$699',
    gift_lifetime: 'MX$2,199',
    perDay: 'MX$1.92',
    savings: '35%',
  },
  PLN: {
    symbol: 'zł',
    monthly: 'zł29.99',
    annual: 'zł199.99',
    family: 'zł399.99',
    lifetime: 'zł799.99',
    gift_3mo: 'zł99.99',
    gift_6mo: 'zł159.99',
    gift_12mo: 'zł199.99',
    gift_lifetime: 'zł599.99',
    perDay: 'zł0.55',
    savings: '30%',
  },
};

// ============================================
// PAYMENT METHODS BY COUNTRY
// ============================================

export function getPaymentMethodsForCountry(countryCode: string): string[] {
  const methods: string[] = ['card']; // Always support cards
  
  // Netherlands - iDEAL is dominant
  if (countryCode === 'NL') {
    methods.push('ideal');
  }
  
  // Belgium - Bancontact is dominant
  if (countryCode === 'BE') {
    methods.push('bancontact');
  }
  
  // Germany - giropay, Sofort, Klarna
  if (countryCode === 'DE') {
    methods.push('giropay', 'sofort', 'klarna');
  }
  
  // Austria - Sofort, Klarna, EPS
  if (countryCode === 'AT') {
    methods.push('sofort', 'klarna', 'eps');
  }
  
  // Poland - P24, BLIK
  if (countryCode === 'PL') {
    methods.push('p24', 'blik');
  }
  
  // Brazil - Boleto, Pix
  if (countryCode === 'BR') {
    methods.push('boleto', 'pix');
  }
  
  // India - UPI
  if (countryCode === 'IN') {
    methods.push('upi');
  }
  
  // Mexico - OXXO
  if (countryCode === 'MX') {
    methods.push('oxxo');
  }
  
  // Japan - Konbini
  if (countryCode === 'JP') {
    methods.push('konbini');
  }
  
  // US - ACH, Cash App, Affirm
  if (countryCode === 'US') {
    methods.push('us_bank_account', 'cashapp', 'affirm');
  }
  
  // UK - Bacs Direct Debit
  if (countryCode === 'GB') {
    methods.push('bacs_debit');
  }
  
  // EU countries - SEPA Direct Debit
  if (EU_COUNTRIES.has(countryCode)) {
    methods.push('sepa_debit');
  }
  
  // Global - Apple Pay, Google Pay (handled by Stripe automatically with 'card')
  // Link (Stripe's one-click checkout) - enabled globally
  methods.push('link');
  
  return methods;
}

// ============================================
// HELPER FUNCTIONS
// ============================================

export function getCurrencyForCountry(countryCode: string): string {
  return COUNTRY_TO_CURRENCY[countryCode] || 'USD';
}

export function isPPPCountry(countryCode: string): boolean {
  return PPP_COUNTRIES.has(countryCode);
}

export function getPriceIdForPlan(
  plan: string, 
  currency: string
): string {
  const currencyPrices = PRICE_IDS[currency] || PRICE_IDS.USD;
  return currencyPrices[plan] || currencyPrices.annual || PRICE_IDS.USD.annual;
}

export function getDisplayPrices(currency: string) {
  return DISPLAY_PRICES[currency] || DISPLAY_PRICES.USD;
}

// Get all currencies we support
export function getSupportedCurrencies(): string[] {
  return Object.keys(DISPLAY_PRICES);
}

// Validate that we have a price ID for a plan/currency combo
export function hasPriceForCurrency(plan: string, currency: string): boolean {
  const prices = PRICE_IDS[currency];
  if (!prices) return false;
  return !!prices[plan];
}

// Fallback logic: if we don't have a price for this currency, use USD
export function getEffectiveCurrency(plan: string, requestedCurrency: string): string {
  if (hasPriceForCurrency(plan, requestedCurrency)) {
    return requestedCurrency;
  }
  console.log(`[Pricing] No ${plan} price for ${requestedCurrency}, falling back to USD`);
  return 'USD';
}
