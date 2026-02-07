// ziggurat-products.ts
// Full Product Audit and Derivatives for Curious Kelly / Lesson of the Day

import type { FinancialInputs } from './types';

// ============================================================
// CURRENT PRODUCT AUDIT - What exists today
// ============================================================

export interface ProductLine {
  id: string;
  name: string;
  category: 'subscription' | 'digital' | 'print' | 'hardware' | 'licensing' | 'services' | 'merchandise';
  status: 'live' | 'beta' | 'development' | 'concept';
  pricing: {
    model: 'subscription' | 'one-time' | 'per-seat' | 'freemium' | 'tiered';
    price: number | { min: number; max: number };
    unit: string;
  };
  description: string;
  targetMarket: string[];
  revenueY1Est: number;
  marginEst: number; // 0-1
  dependencies?: string[];
}

export const CURRENT_PRODUCTS: ProductLine[] = [
  // ============ SUBSCRIPTIONS ============
  {
    id: 'sub-explorer',
    name: 'Explorer (Free)',
    category: 'subscription',
    status: 'live',
    pricing: { model: 'freemium', price: 0, unit: 'forever' },
    description: "Today's lesson only, 1 language, basic Kelly",
    targetMarket: ['casual learners', 'trial users'],
    revenueY1Est: 0,
    marginEst: 0,
  },
  {
    id: 'sub-learner',
    name: 'Learner (Individual)',
    category: 'subscription',
    status: 'live',
    pricing: { model: 'subscription', price: { min: 9.99, max: 199 }, unit: 'month/year/lifetime' },
    description: '365 lessons, all 12 languages, all Kelly archetypes, all ages, offline downloads',
    targetMarket: ['individuals', 'lifelong learners', 'parents'],
    revenueY1Est: 500 * 9.99 * 12 + 200 * 79, // Monthly + Annual Y1
    marginEst: 0.85,
  },
  {
    id: 'sub-family',
    name: 'Family (6 seats)',
    category: 'subscription',
    status: 'live',
    pricing: { model: 'subscription', price: { min: 14.99, max: 349 }, unit: 'month/year/lifetime' },
    description: 'Up to 6 family members, individual profiles, parental controls',
    targetMarket: ['families', 'multi-generational households'],
    revenueY1Est: 100 * 14.99 * 12 + 50 * 119.99,
    marginEst: 0.82,
  },
  {
    id: 'sub-district',
    name: 'School District License',
    category: 'subscription',
    status: 'live',
    pricing: { model: 'per-seat', price: 12, unit: 'student/year' },
    description: 'Bulk licensing for K-12 districts with admin dashboard',
    targetMarket: ['school districts', 'education departments'],
    revenueY1Est: 10 * 5000 * 12, // 10 districts * 5000 students * $12
    marginEst: 0.75,
  },
  
  // ============ AFFILIATE PROGRAM ============
  {
    id: 'affiliate-program',
    name: 'Affiliate Program',
    category: 'services',
    status: 'live',
    pricing: { model: 'tiered', price: { min: 0.20, max: 0.40 }, unit: 'commission rate' },
    description: '5 tiers: Starter (20%) → Advocate (25%) → Ambassador (30%) → Champion (35%) → Legend (40% + 10% recurring)',
    targetMarket: ['influencers', 'educators', 'parents', 'content creators'],
    revenueY1Est: 0, // Cost center, drives subscription revenue
    marginEst: -0.25, // Commission payout
  },
  {
    id: 'language-guardian',
    name: 'Language Guardian Program',
    category: 'services',
    status: 'live',
    pricing: { model: 'tiered', price: 0.50, unit: 'commission for language evangelists' },
    description: 'Translators and language advocates with 50% commission on their language territory',
    targetMarket: ['polyglots', 'native speakers', 'translation professionals'],
    revenueY1Est: 0,
    marginEst: -0.50,
  },
];

// ============================================================
// PRODUCT DERIVATIVES - Daydream expansion
// ============================================================

export const DERIVATIVE_PRODUCTS: ProductLine[] = [
  // ============ DIGITAL DOWNLOADS ============
  {
    id: 'digital-monthly-pack',
    name: 'Monthly Lesson Pack',
    category: 'digital',
    status: 'development',
    pricing: { model: 'one-time', price: 9, unit: 'per month pack' },
    description: '30 daily lessons bundled with offline materials, printables, activities',
    targetMarket: ['homeschoolers', 'parents', 'teachers'],
    revenueY1Est: 500 * 9 * 12,
    marginEst: 0.90,
  },
  {
    id: 'digital-complete-year',
    name: 'Complete Year Bundle',
    category: 'digital',
    status: 'development',
    pricing: { model: 'one-time', price: 29, unit: 'per year bundle' },
    description: '365 lessons with all supplementary materials, one-time purchase',
    targetMarket: ['homeschoolers', 'curriculum buyers'],
    revenueY1Est: 300 * 29,
    marginEst: 0.92,
  },
  {
    id: 'digital-premium-year',
    name: 'Premium Year + Masterclass',
    category: 'digital',
    status: 'development',
    pricing: { model: 'one-time', price: 49, unit: 'per premium bundle' },
    description: 'Complete year + Kelly Masterclass deep dives, expert interviews, extended content',
    targetMarket: ['enthusiasts', 'educators'],
    revenueY1Est: 100 * 49,
    marginEst: 0.88,
  },
  
  // ============ PRINT EDITION ============
  {
    id: 'print-essentials',
    name: 'Print Essentials Kit',
    category: 'print',
    status: 'concept',
    pricing: { model: 'one-time', price: 39, unit: 'per kit' },
    description: 'Quarterly printed lesson cards, activity sheets, poster, offline-first',
    targetMarket: ['low-connectivity families', 'screen-free parents'],
    revenueY1Est: 200 * 39 * 4,
    marginEst: 0.45,
  },
  {
    id: 'print-standard',
    name: 'Kelly Annual Book',
    category: 'print',
    status: 'concept',
    pricing: { model: 'one-time', price: 69, unit: 'per book' },
    description: 'Beautiful hardcover with all 365 lessons, illustrations, QR codes to videos',
    targetMarket: ['gift buyers', 'libraries', 'collectors'],
    revenueY1Est: 300 * 69,
    marginEst: 0.40,
  },
  {
    id: 'print-collectors',
    name: "Collector's Edition",
    category: 'print',
    status: 'concept',
    pricing: { model: 'one-time', price: 149, unit: 'per edition' },
    description: 'Limited edition boxed set with signed prints, exclusive content, lifetime digital access',
    targetMarket: ['collectors', 'donors', 'superfans'],
    revenueY1Est: 100 * 149,
    marginEst: 0.35,
  },
  
  // ============ iLEARN HARDWARE ============
  {
    id: 'hardware-mini',
    name: 'iLearn Mini',
    category: 'hardware',
    status: 'concept',
    pricing: { model: 'one-time', price: 79, unit: 'per device' },
    description: '4" e-ink device preloaded with 365 lessons, solar charging, 2-year battery',
    targetMarket: ['developing markets', 'low-connectivity areas', 'gift market'],
    revenueY1Est: 0, // Y3 launch
    marginEst: 0.25,
    dependencies: ['hardware-division-team'],
  },
  {
    id: 'hardware-standard',
    name: 'iLearn Standard',
    category: 'hardware',
    status: 'concept',
    pricing: { model: 'one-time', price: 199, unit: 'per device' },
    description: '7" color display, WiFi sync, Kelly AI assistant, 5-year content updates',
    targetMarket: ['families', 'schools', 'libraries'],
    revenueY1Est: 0,
    marginEst: 0.30,
    dependencies: ['hardware-division-team'],
  },
  {
    id: 'hardware-pro',
    name: 'iLearn Pro',
    category: 'hardware',
    status: 'concept',
    pricing: { model: 'one-time', price: 399, unit: 'per device' },
    description: '10" premium tablet, cellular, AR capabilities, content creation tools',
    targetMarket: ['educators', 'content creators', 'premium market'],
    revenueY1Est: 0,
    marginEst: 0.35,
    dependencies: ['hardware-division-team'],
  },
  
  // ============ LICENSING ============
  {
    id: 'license-enterprise',
    name: 'Enterprise Content License',
    category: 'licensing',
    status: 'beta',
    pricing: { model: 'per-seat', price: { min: 10000, max: 200000 }, unit: 'annual deal' },
    description: 'White-label Kelly content for corporate learning platforms',
    targetMarket: ['corporate L&D', 'edtech platforms'],
    revenueY1Est: 1 * 50000,
    marginEst: 0.80,
  },
  {
    id: 'license-government',
    name: 'Government Education License',
    category: 'licensing',
    status: 'concept',
    pricing: { model: 'per-seat', price: { min: 50000, max: 500000 }, unit: 'annual contract' },
    description: 'National curriculum integration for education ministries',
    targetMarket: ['education ministries', 'government agencies'],
    revenueY1Est: 0, // Y3 launch
    marginEst: 0.75,
  },
  {
    id: 'license-broadcast',
    name: 'Broadcast License',
    category: 'licensing',
    status: 'concept',
    pricing: { model: 'tiered', price: { min: 25000, max: 150000 }, unit: 'per territory/year' },
    description: 'TV broadcast rights for daily Kelly episodes',
    targetMarket: ['TV networks', 'streaming platforms', 'public broadcasters'],
    revenueY1Est: 2 * 50000,
    marginEst: 0.85,
  },
  
  // ============ LIVE EXPERIENCES ============
  {
    id: 'live-classes',
    name: 'Live Classes with Kelly',
    category: 'services',
    status: 'beta',
    pricing: { model: 'subscription', price: 29, unit: 'per month add-on' },
    description: 'Weekly live interactive sessions with Kelly avatar, Q&A, special guests',
    targetMarket: ['engaged subscribers', 'classrooms'],
    revenueY1Est: 100 * 29 * 12,
    marginEst: 0.70,
  },
  {
    id: 'live-events',
    name: 'Ziggurat Field Trips',
    category: 'services',
    status: 'concept',
    pricing: { model: 'one-time', price: 25, unit: 'per person' },
    description: 'Educational tours of The Ziggurat, production studios, interactive exhibits',
    targetMarket: ['school groups', 'tourists', 'families'],
    revenueY1Est: 10000 * 25,
    marginEst: 0.60,
    dependencies: ['ziggurat-acquisition'],
  },
  {
    id: 'live-camps',
    name: 'Curiosity Camps',
    category: 'services',
    status: 'concept',
    pricing: { model: 'one-time', price: 499, unit: 'per week' },
    description: 'Summer STEM camps at The Ziggurat, hands-on learning with Kelly themes',
    targetMarket: ['families', 'summer camp seekers'],
    revenueY1Est: 200 * 499,
    marginEst: 0.45,
    dependencies: ['ziggurat-acquisition'],
  },
  
  // ============ MERCHANDISE ============
  {
    id: 'merch-plush',
    name: 'Kelly Plush Collection',
    category: 'merchandise',
    status: 'concept',
    pricing: { model: 'one-time', price: { min: 19, max: 49 }, unit: 'per item' },
    description: 'Plush Kelly avatars in all 12 archetypes, limited editions',
    targetMarket: ['kids', 'collectors', 'gift buyers'],
    revenueY1Est: 5000 * 29,
    marginEst: 0.50,
  },
  {
    id: 'merch-apparel',
    name: 'Curious Apparel',
    category: 'merchandise',
    status: 'concept',
    pricing: { model: 'one-time', price: { min: 25, max: 65 }, unit: 'per item' },
    description: 'T-shirts, hoodies with daily lesson themes, archetype designs',
    targetMarket: ['fans', 'gift market'],
    revenueY1Est: 2000 * 35,
    marginEst: 0.45,
  },
  {
    id: 'merch-edu-toys',
    name: 'Kelly Learning Toys',
    category: 'merchandise',
    status: 'concept',
    pricing: { model: 'one-time', price: { min: 29, max: 99 }, unit: 'per toy' },
    description: 'STEM toys, science kits, building sets themed to lessons',
    targetMarket: ['parents', 'gift buyers', 'educators'],
    revenueY1Est: 1000 * 49,
    marginEst: 0.40,
  },
  
  // ============ ADVANCED DERIVATIVES ============
  {
    id: 'api-access',
    name: 'Kelly API',
    category: 'licensing',
    status: 'concept',
    pricing: { model: 'tiered', price: { min: 99, max: 999 }, unit: 'per month' },
    description: 'Developer API for Kelly content, AI, and video generation',
    targetMarket: ['developers', 'edtech startups', 'researchers'],
    revenueY1Est: 50 * 199 * 12,
    marginEst: 0.90,
  },
  {
    id: 'kelly-ai-tutor',
    name: 'Kelly AI Tutor',
    category: 'subscription',
    status: 'concept',
    pricing: { model: 'subscription', price: 19.99, unit: 'per month add-on' },
    description: 'Personalized AI tutoring based on learning progress, Socratic dialogue',
    targetMarket: ['students', 'test prep', 'skill development'],
    revenueY1Est: 200 * 19.99 * 12,
    marginEst: 0.75,
  },
  {
    id: 'certification-program',
    name: 'Kelly Certified Educator',
    category: 'services',
    status: 'concept',
    pricing: { model: 'one-time', price: 299, unit: 'per certification' },
    description: 'Professional development certification for using Kelly in classrooms',
    targetMarket: ['teachers', 'homeschool parents', 'tutors'],
    revenueY1Est: 500 * 299,
    marginEst: 0.80,
  },
  {
    id: 'podcast-sponsorship',
    name: '"Daily Curiosity" Podcast',
    category: 'licensing',
    status: 'concept',
    pricing: { model: 'tiered', price: { min: 1000, max: 10000 }, unit: 'per episode sponsorship' },
    description: 'Audio version of daily lessons with premium sponsorship slots',
    targetMarket: ['podcast listeners', 'commuters', 'audio learners'],
    revenueY1Est: 365 * 2000 * 0.3, // 30% fill rate Y1
    marginEst: 0.85,
  },
];

// ============================================================
// ALL PRODUCTS COMBINED
// ============================================================

export const ALL_PRODUCTS = [...CURRENT_PRODUCTS, ...DERIVATIVE_PRODUCTS];

// ============================================================
// REVENUE PROJECTIONS BY PRODUCT CATEGORY
// ============================================================

export function calculateProductRevenue(inputs: FinancialInputs, year: number, scenario: 'baseline' | 'optimistic' | 'conservative' = 'baseline') {
  const mult = scenario === 'optimistic' ? 1.5 : scenario === 'conservative' ? 0.7 : 1;
  
  const growthFactors = {
    1: 1,
    2: 2.5,
    3: 5,
    4: 9,
    5: 13.5,
  }[year] || 1;
  
  const hw = year >= 3 ? [0, 0, 1, 3, 6][year - 1] : 0;
  
  return {
    subscriptions: {
      individual: (inputs.monthlySubsY1 * inputs.monthlyPrice * 12 + inputs.annualSubsY1 * inputs.annualPrice) * growthFactors * mult,
      family: (100 * 14.99 * 12 + 50 * 119.99) * growthFactors * mult * 0.8,
      district: inputs.districtDealsY1 * inputs.studentsPerDistrict * inputs.pricePerStudent * growthFactors * mult,
    },
    digital: {
      monthlyPack: inputs.monthlyPackY1 * 9 * 12 * growthFactors * mult,
      completeYear: inputs.completeYearY1 * 29 * growthFactors * mult,
      premiumYear: inputs.premiumYearY1 * 49 * growthFactors * mult,
    },
    print: {
      essentials: inputs.essentialsY1 * 39 * 4 * growthFactors * mult,
      standard: inputs.standardY1 * 69 * growthFactors * mult,
      collectors: inputs.collectorsY1 * 149 * growthFactors * mult,
    },
    hardware: {
      mini: inputs.iLearnMiniY3 * 79 * hw * mult,
      standard: inputs.iLearnStdY3 * 199 * hw * mult,
      pro: inputs.iLearnProY3 * 399 * hw * mult,
    },
    licensing: {
      enterprise: inputs.enterpriseDealsY1 * inputs.enterpriseDealSize * Math.pow(1.1, year - 1) * growthFactors * mult * 0.5,
      government: year >= 3 ? inputs.govDealsY3 * inputs.govDealSize * (year - 2) * mult : 0,
      broadcast: 2 * 50000 * growthFactors * mult * 0.3,
    },
    services: {
      liveClasses: 100 * 29 * 12 * growthFactors * mult,
      fieldTrips: year >= 2 ? 10000 * 25 * (year - 1) * mult : 0,
      camps: year >= 2 ? 200 * 499 * (year - 1) * 0.5 * mult : 0,
      certification: 500 * 299 * growthFactors * mult * 0.3,
    },
    merchandise: {
      plush: 5000 * 29 * growthFactors * mult * 0.3,
      apparel: 2000 * 35 * growthFactors * mult * 0.3,
      toys: 1000 * 49 * growthFactors * mult * 0.3,
    },
    newDerivatives: {
      api: 50 * 199 * 12 * growthFactors * mult * 0.2,
      aiTutor: 200 * 19.99 * 12 * growthFactors * mult * 0.3,
      podcast: 365 * 2000 * 0.3 * growthFactors * mult * 0.2,
    },
  };
}

// ============================================================
// TOTAL REVENUE SUMMARY
// ============================================================

export function getTotalRevenue(inputs: FinancialInputs, year: number, scenario: 'baseline' | 'optimistic' | 'conservative' = 'baseline') {
  const rev = calculateProductRevenue(inputs, year, scenario);
  
  const subs = Object.values(rev.subscriptions).reduce((a, b) => a + b, 0);
  const digital = Object.values(rev.digital).reduce((a, b) => a + b, 0);
  const print = Object.values(rev.print).reduce((a, b) => a + b, 0);
  const hardware = Object.values(rev.hardware).reduce((a, b) => a + b, 0);
  const licensing = Object.values(rev.licensing).reduce((a, b) => a + b, 0);
  const services = Object.values(rev.services).reduce((a, b) => a + b, 0);
  const merch = Object.values(rev.merchandise).reduce((a, b) => a + b, 0);
  const newDeriv = Object.values(rev.newDerivatives).reduce((a, b) => a + b, 0);
  
  return {
    byCategory: {
      subscriptions: subs,
      digital,
      print,
      hardware,
      licensing,
      services,
      merchandise: merch,
      newDerivatives: newDeriv,
    },
    total: subs + digital + print + hardware + licensing + services + merch + newDeriv,
  };
}
