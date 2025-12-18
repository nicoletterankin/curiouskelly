import type { VercelRequest, VercelResponse } from '@vercel/node';
import {
  getCurrencyForCountry,
  getDisplayPrices,
  getPriceIdForPlan,
  isPPPCountry,
  PRICE_IDS,
} from './lib/pricing-config';

/**
 * GET /api/price-selector
 * 
 * Lightweight endpoint for client-side price selection.
 * Returns the correct Stripe price ID for a plan/country combo.
 * 
 * Query params:
 * - plan: 'monthly' | 'annual' | 'family' | 'lifetime' | 'gift_3mo' | 'gift_6mo' | 'gift_12mo' | 'gift_lifetime'
 * - country: Optional country code override (detected from headers if not provided)
 * 
 * Response:
 * {
 *   priceId: "price_xxx",
 *   currency: "EUR",
 *   displayPrice: "€49.99",
 *   isPPP: false
 * }
 */
export default async function handler(
  req: VercelRequest,
  res: VercelResponse
) {
  // CORS + Cache
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Cache-Control', 'public, max-age=300'); // Cache 5 min
  
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'GET') return res.status(405).json({ error: 'Method not allowed' });
  
  const plan = req.query.plan as string;
  const validPlans = ['monthly', 'annual', 'family', 'lifetime', 'gift_3mo', 'gift_6mo', 'gift_12mo', 'gift_lifetime'];
  
  if (!plan || !validPlans.includes(plan)) {
    return res.status(400).json({ 
      error: 'invalid_plan',
      message: `Plan must be one of: ${validPlans.join(', ')}`,
    });
  }
  
  // Get country from query or headers
  const country = (
    (req.query.country as string) ||
    (req.headers['x-vercel-ip-country'] as string) ||
    'US'
  ).toUpperCase();
  
  // Get currency for this country
  const currency = getCurrencyForCountry(country);
  
  // Get price ID
  const currencyPrices = PRICE_IDS[currency] || PRICE_IDS.USD;
  const priceId = currencyPrices[plan] || PRICE_IDS.USD[plan];
  
  // Get display price
  const displayPrices = getDisplayPrices(currency);
  const displayPrice = displayPrices[plan as keyof typeof displayPrices] || '';
  
  // Check if PPP
  const isPPP = isPPPCountry(country);
  
  return res.status(200).json({
    plan,
    country,
    currency,
    priceId: priceId || null,
    displayPrice,
    isPPP,
    // Flag if we had to fall back to USD
    usedFallback: !currencyPrices[plan],
  });
}
