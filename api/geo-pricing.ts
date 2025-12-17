import type { VercelRequest, VercelResponse } from '@vercel/node';
import {
  getCurrencyForCountry,
  getDisplayPrices,
  isPPPCountry,
  getPaymentMethodsForCountry,
  PRICE_IDS,
  getSupportedCurrencies,
} from './lib/pricing-config';

/**
 * GET /api/geo-pricing
 * 
 * Returns localized pricing based on user's country.
 * Uses Vercel's geo headers for detection.
 * 
 * Query params:
 * - force_country: Override detected country (for testing)
 * - force_currency: Override currency (for testing)
 * 
 * Response:
 * {
 *   country: "US",
 *   currency: "USD",
 *   prices: { monthly: "$7.99", annual: "$49.99", ... },
 *   priceIds: { monthly: "price_xxx", annual: "price_yyy", ... },
 *   paymentMethods: ["card", "us_bank_account", ...],
 *   isPPP: false,
 *   pppDiscount: null
 * }
 */
export default async function handler(
  req: VercelRequest,
  res: VercelResponse
) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Cache-Control', 'public, max-age=3600'); // Cache for 1 hour
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  try {
    // Get country from Vercel's geo headers
    // Priority: query param > header > default
    const queryCountry = req.query.force_country as string;
    const queryCurrency = req.query.force_currency as string;
    
    const detectedCountry = (
      queryCountry ||
      (req.headers['x-vercel-ip-country'] as string) ||
      'US'
    ).toUpperCase();
    
    // Get currency for this country
    const currency = queryCurrency?.toUpperCase() || getCurrencyForCountry(detectedCountry);
    
    // Get display prices
    const prices = getDisplayPrices(currency);
    
    // Get Stripe price IDs
    const priceIds = PRICE_IDS[currency] || PRICE_IDS.USD;
    
    // Get available payment methods
    const paymentMethods = getPaymentMethodsForCountry(detectedCountry);
    
    // Check if PPP country
    const isPPP = isPPPCountry(detectedCountry);
    
    // PPP discount percentage for display
    const pppDiscounts: Record<string, number> = {
      IN: 50, BR: 40, MX: 35, PL: 30, TR: 45, 
      ID: 55, PH: 45, ZA: 40, AR: 60, CO: 40,
      CL: 30, PK: 55, VN: 50, TH: 35, MY: 30,
    };
    
    const response = {
      country: detectedCountry,
      currency,
      prices,
      priceIds,
      paymentMethods,
      isPPP,
      pppDiscount: isPPP ? pppDiscounts[detectedCountry] || null : null,
      // Metadata
      supportedCurrencies: getSupportedCurrencies(),
      detectedFrom: queryCountry ? 'query' : 'geo',
    };
    
    return res.status(200).json(response);
    
  } catch (error) {
    console.error('[geo-pricing] Error:', error);
    
    // Return USD defaults on error
    return res.status(200).json({
      country: 'US',
      currency: 'USD',
      prices: getDisplayPrices('USD'),
      priceIds: PRICE_IDS.USD,
      paymentMethods: ['card', 'link'],
      isPPP: false,
      pppDiscount: null,
      error: 'Geo detection failed, using defaults',
    });
  }
}
