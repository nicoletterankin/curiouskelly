// ziggurat-data.ts
// All constants and data for Ziggurat Financial Model

import type { 
  BuildingData, 
  FinancialInputs, 
  KellyVision, 
  SDGItem,
  Scenario 
} from './types';

// ============================================================
// BUILDING DATA
// ============================================================

export const BUILDING: BuildingData = {
  name: "Chet Holifield Federal Building",
  nickname: "The Ziggurat",
  address: "24000 Avila Road, Laguna Niguel, CA 92677",
  architect: "William L. Pereira & Associates",
  built: 1971,
  sqft: 1003041,  // Verified from GSA documents
  acres: 92,      // 86.5 main + 5.5 utility
  floors: 7,      // Stepped pyramid: F1-F7
  roofArea: 480000, // SF - potential LED display surface
  terraceArea: 237800, // SF - outdoor space
  
  history: [
    { year: 1968, event: "Construction begins for North American Rockwell Autonetics Division", type: "build" },
    { year: 1971, event: "Completed; Rockwell never occupies", type: "build" },
    { year: 1974, event: "Traded to GSA for 3 surplus facilities", type: "sale" },
    { year: 1975, event: "Featured in 'Death Race 2000' (Roger Corman)", type: "culture" },
    { year: 1978, event: "Renamed for Congressman Chet Holifield", type: "build" },
    { year: 2003, event: "Major roofing renovation (Polyglass)", type: "build" },
    { year: 2005, event: "Elevator system upgraded", type: "build" },
    { year: 2015, event: "Determined NRHP-eligible", type: "build" },
    { year: 2023, event: "First auction - no bids", type: "sale" },
    { year: 2024, event: "Second auction - $177M bid cancelled", type: "sale" },
    { year: 2025, event: "February: IRS vacates, building now vacant", type: "sale" },
    { year: 2025, event: "Third auction - GSA reviewing offers", type: "sale" },
    { year: 2026, event: "Coalition offer submitted, awaiting GSA response", type: "opportunity" },
  ],
  
  // Floor data from GSA documents - 7 levels stepped pyramid
  floorPlan: [
    { level: "F1", name: "Warehouse & Events", sqft: 331200, vision: "Warehouse/Logistics, Event Space, Parking", color: "slate", grade: "Below" },
    { level: "F2", name: "Data Center & Studios", sqft: 252000, vision: "Data Center, Recording Studios, Post-Production", color: "blue", grade: "Partial" },
    { level: "F3", name: "Production & Labs", sqft: 174000, vision: "Content Studios, AI Training Labs, Production", color: "pink", grade: "Above" },
    { level: "F4", name: "MAIN LOBBY ★", sqft: 120000, vision: "MAIN LOBBY (preserved), Kelly Welcome Center, Retail", color: "amber", grade: "At Grade (N)", preserved: true },
    { level: "F5", name: "Engineering", sqft: 70000, vision: "Engineering & Product Teams, Conference Center", color: "cyan", grade: "Above" },
    { level: "F6", name: "Executive", sqft: 40000, vision: "C-Suite, Board Room, VIP Visitor Experience", color: "purple", grade: "Above" },
    { level: "F7", name: "Observatory", sqft: 15841, vision: "Executive Suite, Heliport Operations, Observation Deck", color: "orange", grade: "Above", heliport: true },
  ],
  
  deathRace: {
    year: 1975,
    director: "Paul Bartel",
    producer: "Roger Corman",
    stars: ["David Carradine", "Sylvester Stallone"],
    budget: "$300,000",
    boxOffice: "$5M",
  },
  
  comparables: [
    { name: "Apple Park", sqft: 2800000, cost: "$5B", employees: 12000 },
    { name: "Googleplex", sqft: 2000000, cost: "$1B+", employees: 8000 },
    { name: "Meta MPK", sqft: 1100000, cost: "$400M", employees: 6000 },
    { name: "Khan Academy HQ", sqft: 50000, cost: "$20M", employees: 200 },
    { name: "The Ziggurat", sqft: 1050000, cost: "$213M", employees: 780, highlight: true },
  ],
};

// ============================================================
// KELLY BRANDING VISION
// ============================================================

export const KELLY_VISION: KellyVision = {
  tagline: "Not jumbotrons. Architectural poetry.",
  philosophy: "Honor Pereira's brutalist vision while announcing the building's new purpose. Think Apple Park meets ancient Mesopotamian ziggurat.",
  
  exterior: [
    { item: "Blue LED terrace accents", desc: "Subtle architectural lighting along each stepped level" },
    { item: "Kelly silhouette projection", desc: "East facade at golden hour, visible from I-5" },
    { item: "Brushed aluminum signage", desc: "'CURIOUS KELLY' at ground level, tasteful, timeless" },
    { item: "Solar array with integrated logo", desc: "Kelly visible from aerial/satellite imagery" },
    { item: "Learning gardens", desc: "Landscaping themed to 365 daily lessons" },
  ],
  
  interior: [
    { item: "40-foot Kelly hologram", desc: "The Commons entrance, greeting visitors" },
    { item: "Interactive lesson displays", desc: "Touch screens throughout public spaces" },
    { item: "Subject-themed floors", desc: "Each level reflects its department's focus" },
    { item: "Global learner counter", desc: "Real-time display of active learners worldwide" },
    { item: "Kelly voice wayfinding", desc: "Audio navigation in Kelly's voice" },
  ],
  
  experience: [
    { item: "Sunrise broadcasts", desc: "Daily lesson from Observatory deck at dawn" },
    { item: "School field trips", desc: "Kelly-guided educational tours" },
    { item: "Live class recordings", desc: "Public viewing galleries for production" },
    { item: "'Dinner with Kelly' galas", desc: "Fundraiser events in The Commons" },
    { item: "Wonder Week festival", desc: "Annual celebration of curiosity" },
  ],
};

// ============================================================
// FINANCIAL DEFAULTS - COALITION MODEL ($309M Phase 1)
// From: THE ZIGGURAT Coalition Investment Opportunity (Jan 29, 2026)
// ============================================================

export const DEFAULT_INPUTS: FinancialInputs = {
  // PHASE 1 USES OF CAPITAL ($309M)
  purchasePrice: 200_000_000,         // Pre-auction offer to GSA
  renovationBuildout: 75_000_000,     // Phase 1 renovation (Levels 1-4)
  ledDisplay: 0,                       // Phase 2/3
  techInfrastructure: 4_000_000,      // Initial tech infrastructure
  softCostsContingency: 0,            // Included in reno
  operatingReserve: 30_000_000,       // 18-month operating reserve
  
  // PHASE 1 SOURCES OF CAPITAL ($309M Coalition Invited)
  // PRIs (Program-Related Investments)
  strategicPartnerEquity: 50_000_000,  // Emerson Collective PRI
  institutionalEquity: 50_000_000,     // Chan Zuckerberg Initiative PRI
  historicTaxCredits: 20_000_000,      // 20% of $100M QRE
  debtFinancing: 55_000_000,           // Mission-aligned debt (Heron, Kresge, MacArthur)
  
  // Foundation Grants
  emersonCollective: 50_000_000,       // PRI (counted in strategic)
  czi: 50_000_000,                     // PRI (counted in institutional)
  legoFoundation: 15_000_000,          // Grant - play-based learning
  gatesFoundation: 40_000_000,         // Grant - global education access
  waltonFoundation: 0,                 // Not invited Phase 1
  otherFoundations: 20_000_000,        // Hewlett, Bezos, etc.
  govGrants: 40_000_000,               // NSF, DOE, NEA applications
  missionDebt: 55_000_000,             // CDFIs and mission lenders
  
  // Staffing
  avgSalary: 150_000,
  headcountY1: 90,
  headcountY5: 780,
  
  // Online Subscriptions
  monthlySubsY1: 500,
  monthlyPrice: 9.99,
  annualSubsY1: 200,
  annualPrice: 79,
  districtDealsY1: 10,
  studentsPerDistrict: 5000,
  pricePerStudent: 12,
  
  // Digital Downloads
  monthlyPackY1: 500,
  completeYearY1: 300,
  premiumYearY1: 100,
  
  // Print Edition
  essentialsY1: 200,
  standardY1: 300,
  collectorsY1: 100,
  
  // iLearn Hardware
  iLearnMiniY3: 5000,
  iLearnStdY3: 2000,
  iLearnProY3: 500,
  
  // Licensing
  enterpriseDealsY1: 1,
  enterpriseDealSize: 50000,
  govDealsY3: 3,
  govDealSize: 100000,
  
  // Philanthropic Operating
  philanthroYearlyY1: 15_000_000,
  
  // Operating Costs (Y1 base)
  buildingOps: 4_000_000,
  techInfra: 3_000_000,
  contentProd: 2_000_000,
  marketing: 1_000_000,
  gaLegal: 2_000_000,
};

// ============================================================
// GROWTH MULTIPLIERS
// ============================================================

export const GROWTH = {
  // Revenue growth by year [Y1, Y2, Y3, Y4, Y5]
  subs: [1, 2.5, 5, 9, 13.5],
  district: [1, 3, 6, 9, 12.6],
  digital: [1, 2.5, 5, 7.5, 9.75],
  print: [1, 3, 6, 9, 11.7],
  live: [1, 3, 6, 10, 15],
  hardware: [0, 0, 1, 3, 6],  // Launches Y3
  licensing: [1, 3, 6, 10, 15],
  
  // Cost scaling
  philanthropy: [1, 0.85, 0.72, 0.61, 0.52],  // Declines as product grows
  cogsScale: [1, 0.84, 0.71, 0.62, 0.56],     // Improves with scale
  hwMargin: [0, 0, 0.25, 0.30, 0.35],         // Improves Y3+
  
  // Opex multipliers
  building: [1, 1.1, 1.21, 1.33, 1.4],
  tech: [1, 1.3, 1.63, 1.95, 2.24],
  content: [1, 1.5, 1.95, 2.34, 2.69],
  marketing: [1, 2, 3, 3.9, 4.68],
  ga: [1, 1.2, 1.38, 1.52, 1.67],
  contractors: [1, 1.3, 1.56, 1.79, 1.97],
  
  // Impact metrics - from investor memo 5-year path
  // Daily Learners: 50K -> 500K -> 5M -> 25M -> 100M
  learners: [1, 10, 100, 500, 2000],  // Multiplier on 50K base
  dailyLearnersBase: 50_000,          // Y1 base: 50,000 daily learners
  languages: [5, 12, 24, 36, 47],
  commonsVisitors: [10000, 20000, 35000, 50000, 60000],
  
  // Revenue targets from investor memo (in millions)
  // $1.7M -> $7.8M -> $33M -> $90M -> $202M
  revenueTargets: [1_700_000, 7_800_000, 33_000_000, 90_000_000, 202_000_000],
  
  // EBITDA targets from investor memo
  // ($0.65M) -> ($2.4M) -> ($3M) -> $15M -> $56M
  ebitdaTargets: [-650_000, -2_400_000, -3_000_000, 15_000_000, 56_000_000],
};

// ============================================================
// SCENARIO MULTIPLIERS
// ============================================================

export const SCENARIOS: Record<Scenario, { name: string; mult: number; desc: string }> = {
  baseline: { 
    name: 'Baseline', 
    mult: 1, 
    desc: 'Conservative projections based on market research' 
  },
  optimistic: { 
    name: 'Optimistic', 
    mult: 1.5, 
    desc: 'Strong product-market fit, viral growth' 
  },
  conservative: { 
    name: 'Conservative', 
    mult: 0.7, 
    desc: 'Slower adoption, extended runway needed' 
  },
};

// ============================================================
// LEARNER-AFFILIATE ECONOMIC MODEL
// ============================================================
// Creates economic opportunity for every learner who shares Kelly

export const AFFILIATE_MODEL = {
  // Revenue sharing
  learnerShare: 0.70,           // 70% to learner-affiliates
  platformShare: 0.30,          // 30% to LOTD PBC
  
  // Conversion metrics
  avgConversionRate: 0.05,      // 5% of shares convert
  avgCommissionPerConvert: 2.50, // $2.50 average commission
  
  // Engagement tiers
  tiers: [
    { name: 'Curious', minShares: 0, bonus: 1.0 },
    { name: 'Ambassador', minShares: 10, bonus: 1.1 },
    { name: 'Champion', minShares: 50, bonus: 1.25 },
    { name: 'Legend', minShares: 100, bonus: 1.5 },
  ],
  
  // Economic impact projections
  avgLearnerEarningsY1: 15,     // $15/year average
  avgLearnerEarningsY5: 120,    // $120/year with growth
  
  // The vision: education that pays for itself
  description: 'Unlike Big Tech that extracts value from users, the Learner-Affiliate model returns value to learners. Every share is an opportunity to earn while spreading knowledge.',
};

// ============================================================
// PRODUCT LINES - Square is the DNA
// ============================================================
// Kelly's player is square, Ziggurat is square, iLearn is square
// Square is the new round - ZigTok style

export const PRODUCT_LINES = {
  // Core Digital Products
  kellyApp: {
    name: 'Kelly App',
    type: 'software',
    description: 'Daily 5-minute lessons with 12 teaching styles',
    price: { free: true, premium: 9.99 },
    margin: 0.85,
    launchDate: '2025-Q1',
  },
  
  // Hardware - iLearn Line (Square form factor)
  iLearnMini: {
    name: 'iLearn Mini',
    type: 'hardware',
    description: '7" square tablet optimized for daily lessons',
    price: { retail: 149 },
    cost: 65,
    margin: 0.56,
    launchDate: '2026-Q2',
    formFactor: 'square', // ZigTok style
  },
  iLearnPro: {
    name: 'iLearn Pro',
    type: 'hardware', 
    description: '12" square tablet for classrooms',
    price: { retail: 399 },
    cost: 175,
    margin: 0.56,
    launchDate: '2026-Q4',
    formFactor: 'square',
  },
  iLearnHub: {
    name: 'iLearn Hub',
    type: 'hardware',
    description: 'Family learning station with 4 profiles',
    price: { retail: 599 },
    cost: 280,
    margin: 0.53,
    launchDate: '2027-Q1',
    formFactor: 'square',
  },
  
  // Print Products
  dailyLessonPrint: {
    name: 'The Daily Lesson Print',
    type: 'print',
    description: 'Physical lesson cards, monthly subscription',
    price: { monthly: 14.99 },
    cost: 4.50,
    margin: 0.70,
    launchDate: '2025-Q3',
  },
  annualBook: {
    name: 'Year of Curiosity',
    type: 'print',
    description: '365 lessons in a beautiful hardcover',
    price: { retail: 49.99 },
    cost: 12,
    margin: 0.76,
    launchDate: '2025-Q4',
  },
  
  // Licensing & B2B
  kellyEnterprise: {
    name: 'Kelly Enterprise',
    type: 'licensing',
    description: 'White-label for schools & corporations',
    price: { perSeat: 2.99 },
    margin: 0.90,
    launchDate: '2026-Q1',
  },
};

// ============================================================
// OLYMPICS & MAJOR EVENTS - Growth Catalysts
// ============================================================

export const GROWTH_EVENTS = {
  olympics2028: {
    name: 'LA 2028 Olympics',
    date: '2028-07-14',
    location: 'Los Angeles, CA',
    opportunity: 'Global visibility, Ziggurat as Olympic content partner',
    projectedReach: 5_000_000_000, // 5B viewers
    kellyIntegration: [
      'Olympic lesson series (16 days, 16 sports)',
      'Athlete-hosted lessons',
      'Multi-language broadcast from Ziggurat',
      'iLearn devices in Olympic Village',
    ],
    revenueImpact: {
      subscriptions: 500_000, // New subscribers
      hardware: 100_000,     // iLearn units
      licensing: 10_000_000, // Olympic partnership value
    },
  },
  worldCup2026: {
    name: 'FIFA World Cup 2026',
    date: '2026-06-11',
    location: 'USA, Canada, Mexico',
    opportunity: 'Soccer-themed lessons, Spanish/Portuguese expansion',
    projectedReach: 3_000_000_000,
    kellyIntegration: [
      'Football physics lessons',
      'Geography of host cities',
      'Language learning tie-ins',
    ],
    revenueImpact: {
      subscriptions: 250_000,
      hardware: 50_000,
    },
  },
  superBowl2026: {
    name: 'Super Bowl LX',
    date: '2026-02-08',
    location: 'Santa Clara, CA',
    opportunity: 'First major US event post-launch',
    projectedReach: 120_000_000,
    kellyIntegration: [
      'Game day lessons',
      'Regional launch push',
    ],
    revenueImpact: {
      subscriptions: 50_000,
    },
  },
};

// ============================================================
// 5-YEAR REVENUE PROJECTIONS BY PRODUCT LINE
// ============================================================

export const PRODUCT_REVENUE_PROJECTIONS = {
  year1: {
    kellyApp: { users: 100_000, revenue: 500_000 },
    dailyLessonPrint: { subscribers: 5_000, revenue: 450_000 },
    annualBook: { units: 10_000, revenue: 400_000 },
    total: 1_350_000,
  },
  year2: {
    kellyApp: { users: 500_000, revenue: 3_000_000 },
    dailyLessonPrint: { subscribers: 25_000, revenue: 2_700_000 },
    annualBook: { units: 50_000, revenue: 2_000_000 },
    iLearnMini: { units: 10_000, revenue: 1_490_000 },
    kellyEnterprise: { seats: 50_000, revenue: 1_500_000 },
    total: 10_690_000,
  },
  year3: {
    kellyApp: { users: 2_000_000, revenue: 15_000_000 },
    dailyLessonPrint: { subscribers: 75_000, revenue: 8_100_000 },
    annualBook: { units: 100_000, revenue: 4_000_000 },
    iLearnMini: { units: 50_000, revenue: 7_450_000 },
    iLearnPro: { units: 10_000, revenue: 3_990_000 },
    kellyEnterprise: { seats: 200_000, revenue: 6_000_000 },
    worldCup2026: { bonus: 5_000_000 },
    total: 49_540_000,
  },
  year4: {
    kellyApp: { users: 5_000_000, revenue: 40_000_000 },
    dailyLessonPrint: { subscribers: 150_000, revenue: 16_200_000 },
    annualBook: { units: 200_000, revenue: 8_000_000 },
    iLearnMini: { units: 100_000, revenue: 14_900_000 },
    iLearnPro: { units: 50_000, revenue: 19_950_000 },
    iLearnHub: { units: 20_000, revenue: 11_980_000 },
    kellyEnterprise: { seats: 500_000, revenue: 15_000_000 },
    total: 126_030_000,
  },
  year5: {
    kellyApp: { users: 10_000_000, revenue: 85_000_000 },
    dailyLessonPrint: { subscribers: 300_000, revenue: 32_400_000 },
    annualBook: { units: 400_000, revenue: 16_000_000 },
    iLearnMini: { units: 200_000, revenue: 29_800_000 },
    iLearnPro: { units: 100_000, revenue: 39_900_000 },
    iLearnHub: { units: 50_000, revenue: 29_950_000 },
    kellyEnterprise: { seats: 1_000_000, revenue: 30_000_000 },
    olympics2028: { bonus: 25_000_000 },
    total: 288_050_000,
  },
};

// ============================================================
// IMPACT DATA
// ============================================================

export const SDG_GOALS: SDGItem[] = [
  { 
    goal: 'SDG 4', 
    title: 'Quality Education', 
    desc: 'Ensure inclusive and equitable quality education for all',
    progress: 85 
  },
  { 
    goal: 'SDG 10', 
    title: 'Reduced Inequalities', 
    desc: 'Free access removes economic barriers to learning',
    progress: 70 
  },
  { 
    goal: 'SDG 17', 
    title: 'Partnerships', 
    desc: 'Multi-stakeholder partnerships for sustainable development',
    progress: 60 
  },
];

// ============================================================
// UI CONFIGURATION
// ============================================================

export const TABS = [
  { id: 'vision', label: 'Vision', icon: 'Sparkles' },
  { id: 'building', label: 'Building', icon: 'Building2' },
  { id: 'model', label: 'Model', icon: 'BarChart3' },
  { id: 'product', label: 'Product', icon: 'Package' },
  { id: 'impact', label: 'Impact', icon: 'Heart' },
] as const;

export const COLORS = {
  // Brand
  kelly: '#3B82F6',      // Blue
  accent: '#F59E0B',     // Amber
  
  // Semantic
  success: '#10B981',    // Green
  danger: '#EF4444',     // Red
  warning: '#F59E0B',    // Amber
  info: '#3B82F6',       // Blue
  
  // Chart colors
  chart: {
    online: '#3B82F6',
    digital: '#10B981',
    print: '#F97316',
    live: '#8B5CF6',
    hardware: '#EC4899',
    licensing: '#06B6D4',
    philanthropy: '#F59E0B',
  },
};

// ============================================================
// SLIDER CONFIGURATIONS
// ============================================================

export const SLIDER_CONFIG = {
  purchasePrice: { min: 70_000_000, max: 180_000_000, step: 5_000_000, info: 'GSA minimum $70M, Hilco bid $177M' },
  phase1Reno: { min: 40_000_000, max: 100_000_000, step: 1_000_000, info: 'Commons + Studios, critical systems' },
  phase2Reno: { min: 40_000_000, max: 120_000_000, step: 1_000_000, info: 'Levels 4-7, additional buildout' },
  phase3Reno: { min: 30_000_000, max: 100_000_000, step: 1_000_000, info: 'Levels 8-12, final systems' },
  operatingReserve: { min: 15_000_000, max: 50_000_000, step: 1_000_000, info: '12-18 month runway' },
  
  emersonCollective: { min: 0, max: 50_000_000, step: 5_000_000 },
  czi: { min: 0, max: 50_000_000, step: 5_000_000 },
  legoFoundation: { min: 0, max: 30_000_000, step: 5_000_000 },
  gatesFoundation: { min: 0, max: 50_000_000, step: 5_000_000 },
  waltonFoundation: { min: 0, max: 30_000_000, step: 5_000_000 },
  otherFoundations: { min: 0, max: 30_000_000, step: 5_000_000 },
  historicTaxCredits: { min: 0, max: 40_000_000, step: 5_000_000, info: '20% of qualified rehab expenditure' },
  govGrants: { min: 0, max: 30_000_000, step: 5_000_000, info: 'DOE, EDA, State grants' },
  missionDebt: { min: 0, max: 50_000_000, step: 5_000_000, info: 'CDFI, NMTC below-market financing' },
  
  avgSalary: { min: 100_000, max: 250_000, step: 10_000, info: 'Loaded cost including benefits' },
  headcountY1: { min: 50, max: 200, step: 10 },
  headcountY5: { min: 200, max: 1500, step: 50 },
  
  monthlySubsY1: { min: 100, max: 5000, step: 100 },
  monthlyPrice: { min: 4.99, max: 19.99, step: 1 },
  annualSubsY1: { min: 50, max: 2000, step: 50 },
  annualPrice: { min: 49, max: 149, step: 10 },
  districtDealsY1: { min: 1, max: 50, step: 1 },
  studentsPerDistrict: { min: 1000, max: 30000, step: 1000 },
  pricePerStudent: { min: 5, max: 50, step: 1 },
  
  monthlyPackY1: { min: 100, max: 5000, step: 100, info: '$9 per pack' },
  completeYearY1: { min: 50, max: 3000, step: 50, info: '$29 per year' },
  premiumYearY1: { min: 25, max: 1500, step: 25, info: '$49 per year' },
  
  essentialsY1: { min: 50, max: 2000, step: 50 },
  standardY1: { min: 50, max: 2000, step: 50 },
  collectorsY1: { min: 25, max: 1000, step: 25 },
  
  iLearnMiniY3: { min: 0, max: 50000, step: 1000 },
  iLearnStdY3: { min: 0, max: 20000, step: 500 },
  iLearnProY3: { min: 0, max: 5000, step: 100 },
  
  enterpriseDealsY1: { min: 0, max: 20, step: 1 },
  enterpriseDealSize: { min: 10000, max: 200000, step: 10000, info: 'Grows 10%/year' },
  govDealsY3: { min: 0, max: 20, step: 1 },
  govDealSize: { min: 50000, max: 500000, step: 25000, info: 'Grows 20%/year' },
  
  philanthroYearlyY1: { min: 5_000_000, max: 30_000_000, step: 1_000_000, info: 'Declines ~15%/year' },
  
  buildingOps: { min: 2_000_000, max: 10_000_000, step: 500_000, info: 'Utilities, maintenance, security' },
  techInfra: { min: 1_000_000, max: 10_000_000, step: 500_000, info: 'Cloud, tools, platform' },
  contentProd: { min: 500_000, max: 8_000_000, step: 500_000, info: 'Video, curriculum, localization' },
  marketing: { min: 250_000, max: 10_000_000, step: 250_000, info: 'Scales aggressively early' },
  gaLegal: { min: 1_000_000, max: 8_000_000, step: 500_000 },
};

// ============================================================
// COPY BLOCKS
// ============================================================

export const COPY = {
  hero: {
    tag: "THE VISION",
    title: "A Temple of Learning",
    titleAccent: "for the Digital Age",
    body: "Transform a 1971 brutalist masterpiece into the global headquarters for Curious Kelly, serving 3.7 billion learners without reliable internet through our offline-first educational platform.",
  },
  
  before: {
    title: "Vacant Federal Building",
    subtitle: "BEFORE: January 2026",
    items: [
      "$916M deferred maintenance",
      "All federal tenants relocated",
      "Two failed auction attempts",
      "No preservation plan",
      "At risk of demolition",
    ],
  },
  
  after: {
    title: "Global Education HQ",
    subtitle: "AFTER: 2027+",
    items: [
      "The Commons: Free public learning space",
      "780 full-time jobs in Orange County",
      "Kelly video production studios",
      "Historic preservation via adaptive reuse",
      "iLearn hardware R&D center",
    ],
  },
  
  deathRace: {
    title: "Hollywood History",
    subtitle: "Death Race 2000 (1975)",
    body: "Roger Corman's cult classic used the Ziggurat as the sinister government headquarters of a dystopian America. Starring David Carradine and a young Sylvester Stallone, the film made the building an icon of fictional oppression.",
    cta: "From dystopian government HQ → Global education liberation",
  },
  
  footer: {
    company: "Lesson of the Day, PBC",
    founder: "Nicolette Rankin",
    tagline: "Public Benefit Corporation • California",
    legal: "Open financial model • All calculations live",
  },
};
