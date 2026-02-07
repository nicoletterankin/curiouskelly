// lib/products.ts
// LOTD Financial Model — Data Layer
// All numbers validated February 1, 2026

// ============================================
// DEVICE PORTFOLIO
// ============================================

export const DEVICES = {
  nano: {
    id: 'ilearn-nano',
    name: 'iLearn Nano',
    price: 10,
    cogs: 8,
    margin: 20,
    segment: 'Mission deployment (T4/T5 markets)',
    description: 'Solar/crank powered, e-ink, offline-only',
    tagline: 'Learning without limits',
    ageRange: 'All ages',
    forWhom: 'Villages, refugees, off-grid communities. 1 lesson/day changes lives.',
    color: 'from-emerald-600 to-emerald-800',
    channels: ['NGO', 'Mission'],
    featured: false,
    premium: false,
    specs: ['3" e-ink display', 'Solar + hand crank power', '365 lessons pre-loaded', 'No internet required', 'Waterproof IP67', '10-year battery'],
    inTheBox: ['iLearn Nano', 'Solar panel attachment', 'Hand crank', 'Lanyard', 'Quick start guide (10 languages)'],
    whyThisName: 'Nano means small but mighty. This tiny device carries an entire year of education.',
    successLooksLike: 'A child in a village with no electricity learning their first words in a new language.',
  },
  mini: {
    id: 'ilearn-mini',
    name: 'iLearn Mini',
    price: 99,
    cogs: 55,
    margin: 44,
    segment: 'Young children, ages 2-7',
    description: '7" rugged tablet, parental controls',
    tagline: 'First device, first wonder',
    ageRange: 'Ages 2-7',
    forWhom: 'Toddlers and young children. Drop-proof, parent-controlled, pure learning.',
    color: 'from-sky-500 to-sky-700',
    channels: ['Direct', 'Apple', 'Amazon'],
    featured: false,
    premium: false,
    specs: ['7" IPS display', 'Drop-proof to 4 feet', 'Parental controls built-in', '12-hour battery', 'No ads, no distractions', 'Kelly Plus included (1 year)'],
    inTheBox: ['iLearn Mini', 'Bumper case', 'USB-C charger', 'Parent setup guide', 'Sticker pack'],
    whyThisName: 'Mini for mini humans. Their first dedicated learning device.',
    successLooksLike: 'A 4-year-old teaching their parent what they learned from Kelly today.',
  },
  tiny: {
    id: 'ilearn-tiny',
    name: 'iLearn Tiny',
    price: 249,
    cogs: 125,
    margin: 50,
    segment: 'Students, ages 8-18',
    description: '8" with stylus, Focus Mode, school integration',
    tagline: 'The student companion',
    ageRange: 'Ages 8-18',
    forWhom: 'Students K-12. Focus Mode blocks distractions. Homework help built-in.',
    color: 'from-blue-600 to-indigo-700',
    channels: ['Direct', 'Apple', 'Schools'],
    featured: true,
    premium: false,
    specs: ['8" Retina display', 'Stylus included', 'Focus Mode (no social media)', 'School LMS integration', 'Homework helper AI', 'Kelly Premium included (1 year)'],
    inTheBox: ['iLearn Tiny', 'Precision stylus', 'Protective sleeve', 'USB-C charger', 'School setup guide'],
    whyThisName: 'Tiny but powerful. Everything a student needs, nothing they don\'t.',
    successLooksLike: 'A teenager who finally understands calculus because Kelly explained it their way.',
  },
  pro: {
    id: 'ilearn-pro',
    name: 'iLearn Pro',
    price: 749,
    cogs: 375,
    margin: 50,
    segment: 'Professionals, continuing education',
    description: '10" with keyboard, enterprise SSO, certifications',
    tagline: 'Never stop growing',
    ageRange: 'Adults',
    forWhom: 'Professionals seeking certifications, career changers, lifelong learners.',
    color: 'from-violet-600 to-purple-800',
    channels: ['Direct', 'Enterprise'],
    featured: false,
    premium: false,
    specs: ['10" ProMotion display', 'Detachable keyboard', 'Enterprise SSO', 'Certification tracking', 'Multi-profile support', 'Kelly Premium included (2 years)'],
    inTheBox: ['iLearn Pro', 'Smart keyboard', 'USB-C hub', 'Premium case', 'Enterprise setup guide'],
    whyThisName: 'Pro for professionals who never stop learning.',
    successLooksLike: 'A career changer passing their certification exam on the first try.',
  },
  ultra: {
    id: 'ilearn-ultra',
    name: 'iLearn Ultra',
    price: 4999,
    cogs: 1800,
    margin: 64,
    segment: 'Institutions, enterprise deployment',
    description: '15" 4K workstation, on-device AI, multi-user',
    tagline: 'Fund the mission',
    ageRange: 'Institutions',
    forWhom: 'Every Ultra sold funds 160 Nanos. Premium hardware, maximum impact.',
    color: 'from-amber-500 to-orange-600',
    channels: ['Direct', 'Enterprise'],
    featured: false,
    premium: true,
    specs: ['15" 4K OLED display', 'On-device AI processing', 'Multi-user profiles (up to 50)', 'Enterprise deployment tools', 'White-glove setup included', 'Kelly Premium included (lifetime)'],
    inTheBox: ['iLearn Ultra', 'Precision keyboard', 'Thunderbolt dock', 'Premium rolling case', 'Dedicated support contact', 'Certificate: 160 Nanos funded'],
    whyThisName: 'Ultra premium, ultra impact. The flagship that funds the mission.',
    successLooksLike: 'A corporate training room where employees actually look forward to learning.',
  },
} as const;

// ============================================
// HARDWARE UNITS BY YEAR (actual units)
// ============================================

export const HARDWARE_UNITS = [
  { year: 2026, nano: 0,        mini: 2000,    tiny: 2500,    pro: 500,     ultra: 50,    total: 5050 },
  { year: 2027, nano: 50000,    mini: 10000,   tiny: 10000,   pro: 2000,    ultra: 200,   total: 72200 },
  { year: 2028, nano: 500000,   mini: 50000,   tiny: 40000,   pro: 8000,    ultra: 500,   total: 598500 },
  { year: 2029, nano: 2000000,  mini: 150000,  tiny: 100000,  pro: 20000,   ultra: 1500,  total: 2271500 },
  { year: 2030, nano: 5000000,  mini: 300000,  tiny: 200000,  pro: 50000,   ultra: 5000,  total: 5555000 },
  { year: 2031, nano: 10000000, mini: 500000,  tiny: 350000,  pro: 80000,   ultra: 10000, total: 10940000 },
  { year: 2032, nano: 20000000, mini: 800000,  tiny: 500000,  pro: 120000,  ultra: 20000, total: 21440000 },
] as const;

// ============================================
// HARDWARE UNITS FOR CHARTS (in thousands)
// ============================================

export const HARDWARE_UNITS_K = [
  { year: 2026, nano: 0,     mini: 2,   tiny: 2.5,  pro: 0.5,  ultra: 0.05 },
  { year: 2027, nano: 50,    mini: 10,  tiny: 10,   pro: 2,    ultra: 0.2 },
  { year: 2028, nano: 500,   mini: 50,  tiny: 40,   pro: 8,    ultra: 0.5 },
  { year: 2029, nano: 2000,  mini: 150, tiny: 100,  pro: 20,   ultra: 1.5 },
  { year: 2030, nano: 5000,  mini: 300, tiny: 200,  pro: 50,   ultra: 5 },
  { year: 2031, nano: 10000, mini: 500, tiny: 350,  pro: 80,   ultra: 10 },
  { year: 2032, nano: 20000, mini: 800, tiny: 500,  pro: 120,  ultra: 20 },
] as const;

// ============================================
// HARDWARE REVENUE ($M)
// ============================================

export const HARDWARE_REVENUE = [
  { year: 2026, nano: 0,   mini: 0.2,  tiny: 0.6,  pro: 0.4,  ultra: 0.25, total: 1.4 },
  { year: 2027, nano: 0.5, mini: 1,    tiny: 2.5,  pro: 1.5,  ultra: 1,    total: 6.5 },
  { year: 2028, nano: 5,   mini: 5,    tiny: 10,   pro: 6,    ultra: 2.5,  total: 28.5 },
  { year: 2029, nano: 20,  mini: 15,   tiny: 25,   pro: 15,   ultra: 7.5,  total: 82.5 },
  { year: 2030, nano: 50,  mini: 30,   tiny: 50,   pro: 37.5, ultra: 25,   total: 192.5 },
  { year: 2031, nano: 100, mini: 50,   tiny: 87,   pro: 60,   ultra: 50,   total: 347 },
  { year: 2032, nano: 200, mini: 79,   tiny: 125,  pro: 90,   ultra: 100,  total: 594 },
] as const;

// ============================================
// FINANCIAL PROJECTIONS (Investment Ready Model)
// Canonical numbers from $3B Coalition Capital Formation
// ============================================

export const FINANCIAL_PROJECTIONS = [
  { year: 2026, revenue: 22,    ebitda: -65,   margin: -295, learners: 15000000,   access: 15000000 },
  { year: 2027, revenue: 125,   ebitda: -60,   margin: -48,  learners: 75000000,   access: 75000000 },
  { year: 2028, revenue: 430,   ebitda: 40,    margin: 9,    learners: 280000000,  access: 280000000 },
  { year: 2029, revenue: 1280,  ebitda: 510,   margin: 40,   learners: 700000000,  access: 700000000 },
  { year: 2030, revenue: 3300,  ebitda: 1800,  margin: 55,   learners: 1400000000, access: 1400000000 },
  { year: 2032, revenue: 15300, ebitda: 10300, margin: 67,   learners: 3500000000, access: 3500000000 },
  { year: 2035, revenue: 45000, ebitda: 32000, margin: 71,   learners: 8000000000, access: 8000000000 },
] as const;

// ============================================
// REVENUE BY PRODUCT LINE ($M)
// ============================================

export const REVENUE_BY_LINE = [
  { year: 2026, broadcast: 2,   hardware: 1.4,   print: 0.5, infrastructure: 0,   total: 3.9 },
  { year: 2027, broadcast: 8,   hardware: 6.5,   print: 2,   infrastructure: 1,   total: 17.5 },
  { year: 2028, broadcast: 25,  hardware: 28.5,  print: 5,   infrastructure: 7,   total: 65.5 },
  { year: 2029, broadcast: 60,  hardware: 82.5,  print: 12,  infrastructure: 18,  total: 172.5 },
  { year: 2030, broadcast: 120, hardware: 192.5, print: 25,  infrastructure: 40,  total: 377.5 },
  { year: 2031, broadcast: 200, hardware: 347,   print: 40,  infrastructure: 75,  total: 662 },
  { year: 2032, broadcast: 300, hardware: 594,   print: 60,  infrastructure: 140, total: 1094 },
] as const;

// ============================================
// LEARNER ADOPTION (millions)
// ============================================

export const LEARNER_ADOPTION = [
  { year: 2026, total: 0.1,  t1: 0.05, t2: 0.03,  t3: 0.015, t4: 0.005, t5: 0 },
  { year: 2027, total: 1,    t1: 0.4,  t2: 0.3,   t3: 0.2,   t4: 0.08,  t5: 0.02 },
  { year: 2028, total: 10,   t1: 3,    t2: 3.5,   t3: 2.5,   t4: 0.8,   t5: 0.2 },
  { year: 2029, total: 50,   t1: 12,   t2: 18,    t3: 14,    t4: 5,     t5: 1 },
  { year: 2030, total: 150,  t1: 30,   t2: 50,    t3: 45,    t4: 20,    t5: 5 },
  { year: 2031, total: 300,  t1: 50,   t2: 90,    t3: 100,   t4: 45,    t5: 15 },
  { year: 2032, total: 500,  t1: 70,   t2: 140,   t3: 170,   t4: 90,    t5: 30 },
] as const;

// ============================================
// TERRITORY TIERS
// ============================================

export const TERRITORY_TIERS = [
  { tier: 1, gdp: '>$40K',   population: '1.2B', description: 'Premium markets',      pct2032: 14 },
  { tier: 2, gdp: '$15-40K', population: '2.5B', description: 'Developing economies', pct2032: 28 },
  { tier: 3, gdp: '$5-15K',  population: '2.8B', description: 'Growth markets',       pct2032: 34 },
  { tier: 4, gdp: '$1.5-5K', population: '1.5B', description: 'Emerging markets',     pct2032: 18 },
  { tier: 5, gdp: '<$1.5K',  population: '1.0B', description: 'Subsidized',           pct2032: 6 },
] as const;

// ============================================
// LTV BY SEGMENT
// ============================================

export const LTV_BY_SEGMENT = [
  { segment: 'T1 Student (age 6)',   ltv: 3251, cac: 8,  ratio: 406, payback: '1 mo' },
  { segment: 'T1 Child (age 2)',     ltv: 2847, cac: 8,  ratio: 356, payback: '1 mo' },
  { segment: 'T1 Teen (age 12)',     ltv: 2912, cac: 10, ratio: 291, payback: '1 mo' },
  { segment: 'T1 Adult (age 25)',    ltv: 2567, cac: 25, ratio: 103, payback: '2 mo' },
  { segment: 'T1 Professional (35)', ltv: 2891, cac: 50, ratio: 58,  payback: '3 mo' },
  { segment: 'T1 Retiree (age 65)',  ltv: 1842, cac: 15, ratio: 123, payback: '2 mo' },
  { segment: 'T2 Student',           ltv: 878,  cac: 4,  ratio: 220, payback: '1 mo' },
  { segment: 'T2 Adult',             ltv: 693,  cac: 12, ratio: 58,  payback: '2 mo' },
  { segment: 'T3 Student',           ltv: 325,  cac: 2,  ratio: 163, payback: '1 mo' },
  { segment: 'T3 Adult',             ltv: 257,  cac: 5,  ratio: 51,  payback: '2 mo' },
  { segment: 'T4 (Subsidized)',      ltv: 98,   cac: 1,  ratio: 98,  payback: '1 mo' },
  { segment: 'T5 (Paid to Learn)',   ltv: 16,   cac: 0,  ratio: Infinity, payback: '0' },
] as const;

// ============================================
// LTV BY AGE (for chart)
// ============================================

export const LTV_BY_AGE = [
  { age: 2,  t1: 2847, t2: 769, t3: 285 },
  { age: 6,  t1: 3251, t2: 878, t3: 325 },
  { age: 12, t1: 2912, t2: 786, t3: 291 },
  { age: 18, t1: 2398, t2: 647, t3: 240 },
  { age: 25, t1: 2567, t2: 693, t3: 257 },
  { age: 35, t1: 2891, t2: 781, t3: 289 },
  { age: 50, t1: 2634, t2: 711, t3: 263 },
  { age: 65, t1: 1842, t2: 497, t3: 184 },
] as const;

// ============================================
// CROSS-SUBSIDY MODEL
// ============================================

export const CROSS_SUBSIDY = {
  ultraPrice: 4999,
  ultraCogs: 1800,
  ultraProfit: 3199,
  nanoDeploymentCost: 20,  // COGS $8 + shipping $5 + distribution $5 + infrastructure $2
  nanosPerUltra: 159,      // 3199 / 20 = 159.95
  
  byYear: [
    { year: 2026, ultras: 50,    profit: 159950,   nanosEnabled: 7950 },
    { year: 2027, ultras: 200,   profit: 639800,   nanosEnabled: 31800 },
    { year: 2028, ultras: 500,   profit: 1599500,  nanosEnabled: 79500 },
    { year: 2029, ultras: 1500,  profit: 4798500,  nanosEnabled: 238500 },
    { year: 2030, ultras: 5000,  profit: 15995000, nanosEnabled: 795000 },
    { year: 2031, ultras: 10000, profit: 31990000, nanosEnabled: 1590000 },
    { year: 2032, ultras: 20000, profit: 63980000, nanosEnabled: 3180000 },
  ],
} as const;

// ============================================
// KEY DATES
// ============================================

export const KEY_DATES = [
  { date: 'Feb 1, 2026',  event: 'GSA Third Auction Opens' },
  { date: 'Feb 7, 2026',  event: 'Target: Ziggurat Bid Accepted' },
  { date: 'Jun 2026',     event: 'Escrow Closes' },
  { date: 'Dec 2026',     event: 'Kelly Launches from Ziggurat' },
  { date: 'Q4 2027',      event: 'iLearn in Apple Stores (100 locations)' },
  { date: 'Jul 2028',     event: 'LA2028 Olympics — Education Scoreboard' },
  { date: '2028',         event: 'Breakeven' },
  { date: '2030',         event: '150M Learners, $378M Revenue' },
  { date: '2032',         event: '500M Learners, $1.09B Revenue' },
  { date: '2075',         event: 'SDG 4 Achieved: 8B Learners' },
] as const;

// ============================================
// CAPITAL STRUCTURE
// ============================================

export const CAPITAL_PHASES = [
  { phase: 1, years: '2026',      amount: 309000000,  purpose: 'Ziggurat Acquisition + Launch' },
  { phase: 2, years: '2027-2028', amount: 500000000,  purpose: 'Full Buildout + Manufacturing' },
  { phase: 3, years: '2028-2030', amount: 800000000,  purpose: 'International (100 languages)' },
  { phase: 4, years: '2030-2032', amount: 1400000000, purpose: 'Scale (500M learners)' },
] as const;

export const TOTAL_CAPITAL = 3009000000; // $3.009B

// ============================================
// GRAYSCALE PALETTE
// ============================================

export const GRAY = {
  900: '#111111',
  800: '#1f1f1f',
  700: '#333333',
  600: '#4a4a4a',
  500: '#666666',
  400: '#888888',
  300: '#aaaaaa',
  200: '#cccccc',
  100: '#e8e8e8',
  50:  '#f5f5f5',
} as const;

// ============================================
// HELPER FUNCTIONS
// ============================================

export function formatCurrency(amount: number): string {
  if (amount >= 1_000_000_000) return `$${(amount / 1_000_000_000).toFixed(2).replace(/\.?0+$/, '')}B`;
  if (amount >= 1_000_000) return `$${(amount / 1_000_000).toFixed(1).replace(/\.?0+$/, '')}M`;
  if (amount >= 1_000) return `$${(amount / 1_000).toFixed(0)}K`;
  return `$${amount.toFixed(2)}`;
}

export function formatNumber(num: number): string {
  if (num >= 1_000_000_000) return `${(num / 1_000_000_000).toFixed(1).replace(/\.?0+$/, '')}B`;
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(1).replace(/\.?0+$/, '')}M`;
  if (num >= 1_000) return `${(num / 1_000).toFixed(0)}K`;
  return num.toLocaleString();
}

// ============================================
// KELLY SUBSCRIPTIONS
// ============================================

export const KELLY_SUBSCRIPTIONS = {
  free: {
    id: 'kelly-free',
    name: 'Free',
    tagline: 'Start learning today',
    price: 0,
    interval: null,
    annualPrice: 0,
    features: ['1 lesson per week', 'Ad-supported', 'Basic progress tracking'],
    cta: 'Get Started',
    highlighted: false,
    target2032Subs: 280_000_000,
    target2032Revenue: 0,
  },
  basic: {
    id: 'kelly-basic',
    name: 'Basic',
    tagline: 'Daily learning, no ads',
    price: 4,
    interval: 'month',
    annualPrice: 40,
    features: ['Daily lessons', 'Ad-free experience', 'Progress tracking', 'Offline download (5 lessons)'],
    cta: 'Start Basic',
    highlighted: false,
    target2032Subs: 12_000_000,
    target2032Revenue: 576_000_000,
  },
  plus: {
    id: 'kelly-plus',
    name: 'Plus',
    tagline: 'Live classes included',
    price: 10,
    interval: 'month',
    annualPrice: 100,
    features: ['Everything in Basic', '5 live classes/month', 'Full 7-year library', 'All 12 archetypes', 'Unlimited offline'],
    cta: 'Go Plus',
    highlighted: true,
    target2032Subs: 6_000_000,
    target2032Revenue: 720_000_000,
  },
  premium: {
    id: 'kelly-premium',
    name: 'Premium',
    tagline: 'Unlimited everything',
    price: 25,
    interval: 'month',
    annualPrice: 250,
    features: ['Everything in Plus', 'Unlimited live classes', 'Family sharing (4 seats)', 'Priority support', 'Certification tracks'],
    cta: 'Go Premium',
    highlighted: false,
    target2032Subs: 1_500_000,
    target2032Revenue: 450_000_000,
  },
  institution: {
    id: 'kelly-institution',
    name: 'Institution',
    tagline: 'For schools and organizations',
    price: 500,
    interval: 'month',
    annualPrice: 5000,
    features: ['30 seats included', 'Admin dashboard', 'LMS integration', 'Custom content', 'Dedicated support', 'Analytics & reporting'],
    cta: 'Contact Sales',
    highlighted: false,
    target2032Subs: 50_000,
    target2032Revenue: 300_000_000,
  },
} as const;

// Alias for backward compatibility
export const SUBSCRIPTION_TIERS = KELLY_SUBSCRIPTIONS;

// ============================================
// COALITION TIERS
// ============================================

export const COALITION_TIERS = {
  founding: { id: 'founding', name: 'Founding', min: 100_000_000, emoji: '🏛️' },
  anchor: { id: 'anchor', name: 'Anchor', min: 25_000_000, max: 99_000_000, emoji: '⚓' },
  builder: { id: 'builder', name: 'Builder', min: 1_000_000, max: 24_000_000, emoji: '🔨' },
  pioneer: { id: 'pioneer', name: 'Pioneer', min: 10_000, max: 999_000, emoji: '🚀' },
} as const;

// Helper to get tier by ID
export function getTierById(tierId: string) {
  // Check coalition tiers
  const coalitionTier = Object.values(COALITION_TIERS).find(t => t.id === tierId);
  if (coalitionTier) return coalitionTier;
  
  // Check subscription tiers
  const subTier = Object.values(KELLY_SUBSCRIPTIONS).find(t => t.id === tierId);
  if (subTier) return subTier;
  
  // Check iLearn devices
  const device = Object.values(DEVICES).find(d => d.id === tierId);
  if (device) return device;
  
  return null;
}

// ============================================
// ILEARN ALIASES (backward compatibility)
// ============================================

export const ILEARN_DEVICES = DEVICES;

export const ILEARN_BUNDLES = {
  starter: {
    id: 'ilearn-starter-bundle',
    name: 'Family Starter',
    tagline: 'Perfect first step',
    price: 149,
    savingsPercent: 10,
    perUnit: 75,
    featured: false,
    contents: ['1x iLearn Mini', '1x iLearn Nano', 'Family sharing setup'],
  },
  family: {
    id: 'ilearn-family-bundle',
    name: 'Family Complete',
    tagline: 'Learning for everyone',
    price: 449,
    savingsPercent: 15,
    perUnit: 112,
    featured: true,
    contents: ['2x iLearn Mini', '1x iLearn Tiny', '1x iLearn Nano', 'Family dashboard'],
  },
  classroom: {
    id: 'ilearn-classroom-bundle',
    name: 'Classroom Pack',
    tagline: '30 students, one price',
    price: 2499,
    savingsPercent: 20,
    perUnit: 83,
    featured: false,
    contents: ['25x iLearn Nano', '5x iLearn Mini', 'Teacher dashboard', 'LMS integration'],
  },
  enterprise: {
    id: 'ilearn-enterprise-bundle',
    name: 'Enterprise Suite',
    tagline: 'Organization-wide learning',
    price: 9999,
    savingsPercent: 25,
    perUnit: 200,
    featured: false,
    contents: ['1x iLearn Ultra', '10x iLearn Pro', '20x iLearn Tiny', 'Dedicated support', 'Custom deployment'],
  },
} as const;
