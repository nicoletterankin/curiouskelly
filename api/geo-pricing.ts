/**
 * Geo-Pricing API
 * GET /api/geo-pricing
 * 
 * Returns localized pricing based on user's country.
 * Uses Purchasing Power Parity (PPP) for fair global pricing.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

interface PricingData {
  country: string;
  currency: string;
  symbol: string;
  prices: {
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
  };
  isPPP: boolean;
}

// PPP-adjusted pricing by region
const PRICING_TIERS: Record<string, PricingData> = {
  // Tier 1: Full price (US, Canada, UK, Australia, etc.)
  US: {
    country: 'US',
    currency: 'USD',
    symbol: '$',
    prices: {
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
    isPPP: false,
  },

  // UK pricing in GBP
  GB: {
    country: 'GB',
    currency: 'GBP',
    symbol: '£',
    prices: {
      monthly: '£6.49',
      annual: '£39.99',
      family: '£79.99',
      lifetime: '£159.99',
      gift_3mo: '£19.99',
      gift_6mo: '£34.99',
      gift_12mo: '£39.99',
      gift_lifetime: '£119.99',
      perDay: '£0.11',
      savings: '48%',
    },
    isPPP: false,
  },

  // EU pricing in EUR
  EU: {
    country: 'EU',
    currency: 'EUR',
    symbol: '€',
    prices: {
      monthly: '€7.49',
      annual: '€46.99',
      family: '€93.99',
      lifetime: '€189.99',
      gift_3mo: '€22.99',
      gift_6mo: '€37.99',
      gift_12mo: '€46.99',
      gift_lifetime: '€139.99',
      perDay: '€0.13',
      savings: '48%',
    },
    isPPP: false,
  },

  // India (PPP adjusted)
  IN: {
    country: 'IN',
    currency: 'INR',
    symbol: '₹',
    prices: {
      monthly: '₹199',
      annual: '₹999',
      family: '₹1,999',
      lifetime: '₹3,999',
      gift_3mo: '₹499',
      gift_6mo: '₹799',
      gift_12mo: '₹999',
      gift_lifetime: '₹2,999',
      perDay: '₹2.74',
      savings: '58%',
    },
    isPPP: true,
  },

  // Brazil (PPP adjusted)
  BR: {
    country: 'BR',
    currency: 'BRL',
    symbol: 'R$',
    prices: {
      monthly: 'R$24.90',
      annual: 'R$149.90',
      family: 'R$299.90',
      lifetime: 'R$599.90',
      gift_3mo: 'R$74.90',
      gift_6mo: 'R$124.90',
      gift_12mo: 'R$149.90',
      gift_lifetime: 'R$449.90',
      perDay: 'R$0.41',
      savings: '50%',
    },
    isPPP: true,
  },

  // Mexico (PPP adjusted)
  MX: {
    country: 'MX',
    currency: 'MXN',
    symbol: '$',
    prices: {
      monthly: '$99 MXN',
      annual: '$599 MXN',
      family: '$1,199 MXN',
      lifetime: '$2,399 MXN',
      gift_3mo: '$299 MXN',
      gift_6mo: '$499 MXN',
      gift_12mo: '$599 MXN',
      gift_lifetime: '$1,799 MXN',
      perDay: '$1.64 MXN',
      savings: '50%',
    },
    isPPP: true,
  },
};

// EU countries
const EU_COUNTRIES = new Set([
  'DE', 'FR', 'IT', 'ES', 'NL', 'BE', 'AT', 'PT', 'IE', 'FI',
  'GR', 'SK', 'SI', 'LU', 'EE', 'LV', 'LT', 'MT', 'CY', 'HR'
]);

// Full-price countries (same as US)
const TIER1_COUNTRIES = new Set([
  'US', 'CA', 'AU', 'NZ', 'SG', 'HK', 'JP', 'KR', 'CH', 'NO', 'SE', 'DK'
]);

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Cache-Control', 'public, max-age=3600'); // 1 hour cache

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Get country from Vercel headers
    const country = (req.headers['x-vercel-ip-country'] as string) || 'US';
    const forceCountry = req.query.country as string;
    const effectiveCountry = forceCountry?.toUpperCase() || country.toUpperCase();

    // Determine pricing tier
    let pricing: PricingData;

    if (PRICING_TIERS[effectiveCountry]) {
      pricing = PRICING_TIERS[effectiveCountry];
    } else if (EU_COUNTRIES.has(effectiveCountry)) {
      pricing = { ...PRICING_TIERS.EU, country: effectiveCountry };
    } else if (TIER1_COUNTRIES.has(effectiveCountry)) {
      pricing = { ...PRICING_TIERS.US, country: effectiveCountry };
    } else {
      // Default to US pricing
      pricing = { ...PRICING_TIERS.US, country: effectiveCountry };
    }

    return res.status(200).json(pricing);

  } catch (error) {
    console.error('[geo-pricing] Error:', error);
    return res.status(200).json(PRICING_TIERS.US);
  }
}
