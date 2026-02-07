// ziggurat-types.ts
// TypeScript interfaces for Ziggurat Financial Model

export type Scenario = 'baseline' | 'optimistic' | 'conservative';

export interface HistoryEvent {
  year: number;
  event: string;
  type: 'build' | 'sale' | 'legal' | 'culture' | 'opportunity';
}

export interface Floor {
  level: string;
  name: string;
  sqft: number;
  vision: string;
  color: 'amber' | 'pink' | 'blue' | 'purple' | 'emerald' | 'cyan' | 'violet' | 'orange';
}

export interface Comparable {
  name: string;
  sqft: number;
  cost: string;
  employees: number;
  highlight?: boolean;
}

export interface DeathRaceData {
  year: number;
  director: string;
  producer: string;
  stars: string[];
  budget: string;
  boxOffice: string;
}

export interface BuildingData {
  name: string;
  nickname: string;
  address: string;
  architect: string;
  built: number;
  sqft: number;
  acres: number;
  floors: number;
  history: HistoryEvent[];
  floorPlan: Floor[];
  deathRace: DeathRaceData;
  comparables: Comparable[];
}

export interface FinancialInputs {
  // Infrastructure ($700M-$925M total per pitch)
  purchasePrice: number;              // $150-250M acquisition
  renovationBuildout: number;         // $350-400M renovation & buildout  
  ledDisplay: number;                 // $50-75M LED display system
  techInfrastructure: number;         // $75-100M technology
  softCostsContingency: number;       // $75-100M soft costs & contingency
  operatingReserve: number;           // $25M reserve
  
  // Capital Sources (aligned with pitch capital structure)
  strategicPartnerEquity: number;     // 40% - tech/education partner
  institutionalEquity: number;        // 30% - impact investors
  historicTaxCredits: number;         // 10% - 20% of QRE
  debtFinancing: number;              // 20% - construction/permanent
  
  // Legacy foundation fields (backwards compat)
  emersonCollective: number;
  czi: number;
  legoFoundation: number;
  gatesFoundation: number;
  waltonFoundation: number;
  otherFoundations: number;
  govGrants: number;
  missionDebt: number;
  
  // Staffing
  avgSalary: number;
  headcountY1: number;
  headcountY5: number;
  
  // Online Subscriptions
  monthlySubsY1: number;
  monthlyPrice: number;
  annualSubsY1: number;
  annualPrice: number;
  districtDealsY1: number;
  studentsPerDistrict: number;
  pricePerStudent: number;
  
  // Digital Downloads
  monthlyPackY1: number;
  completeYearY1: number;
  premiumYearY1: number;
  
  // Print Edition
  essentialsY1: number;
  standardY1: number;
  collectorsY1: number;
  
  // iLearn Hardware
  iLearnMiniY3: number;
  iLearnStdY3: number;
  iLearnProY3: number;
  
  // Licensing
  enterpriseDealsY1: number;
  enterpriseDealSize: number;
  govDealsY3: number;
  govDealSize: number;
  
  // Philanthropic Operating
  philanthroYearlyY1: number;
  
  // Operating Costs (Y1 base)
  buildingOps: number;
  techInfra: number;
  contentProd: number;
  marketing: number;
  gaLegal: number;
}

export interface YearCalculation {
  year: number;
  headcount: number;
  
  // Revenue
  onlineTotal: number;
  digitalRev: number;
  printRev: number;
  liveRev: number;
  hwRev: number;
  licenseRev: number;
  totalProductRev: number;
  philanthroOperating: number;
  totalRevenue: number;
  
  // Costs
  totalCOR: number;
  grossProfit: number;
  payroll: number;
  contractors: number;
  totalOpex: number;
  
  // Results
  netOperating: number;
  selfSustaining: number;
  
  // Metrics
  grossMargin: number;
  philanthroDependency: number;
  
  // Impact
  dailyActiveLearners: number;
  languages: number;
  commonsVisitors: number;
  cumDistricts: number;
}

export interface CapitalCalculation {
  philanthropyTotal: number;
  acquisitionTotal: number;
  totalCapitalNeeded: number;
  totalCapitalRaised: number;
  fundingGap: number;
  fundingPercent: number;
  totalRenovation: number;
}

export interface FullCalculation extends CapitalCalculation {
  years: YearCalculation[];
  current: YearCalculation;
  breakEvenYear: number | null;
  fiveYearProductRev: number;
  fiveYearPhilanthropy: number;
}

export interface FeedbackForm {
  name: string;
  email: string;
  org: string;
  type: 'question' | 'suggestion' | 'correction' | 'investment' | 'partnership';
  message: string;
  submitted: boolean;
}

export interface KellyVisionItem {
  item: string;
  desc: string;
}

export interface KellyVision {
  tagline: string;
  philosophy: string;
  exterior: KellyVisionItem[];
  interior: KellyVisionItem[];
  experience: KellyVisionItem[];
}

export interface SDGItem {
  goal: string;
  title: string;
  desc: string;
  progress: number;
}

export interface RevenueSegment {
  label: string;
  value: number;
  color: string;
}

// Product Lines (Square DNA: Kelly player, Ziggurat, iLearn)
export interface ProductLine {
  name: string;
  type: 'software' | 'hardware' | 'print' | 'licensing';
  description: string;
  price: { free?: boolean; premium?: number; retail?: number; monthly?: number; perSeat?: number };
  cost?: number;
  margin: number;
  launchDate: string;
  formFactor?: 'square';
}

// Growth Events (Olympics, World Cup, etc.)
export interface GrowthEvent {
  name: string;
  date: string;
  location: string;
  opportunity: string;
  projectedReach: number;
  kellyIntegration: string[];
  revenueImpact: {
    subscriptions?: number;
    hardware?: number;
    licensing?: number;
  };
}

// Learner-Affiliate Model
export interface AffiliateModel {
  learnerShare: number;
  platformShare: number;
  avgConversionRate: number;
  avgCommissionPerConvert: number;
  tiers: Array<{ name: string; minShares: number; bonus: number }>;
  avgLearnerEarningsY1: number;
  avgLearnerEarningsY5: number;
  description: string;
}

// Investor Workspace
export interface InvestorModel {
  id: string;
  investor_name: string;
  investor_email: string;
  organization: string;
  notes: string;
  is_public: boolean;
  model_inputs: Record<string, number>;
  view_count: number;
  created_at: string;
  updated_at: string;
}

export interface InvestorQuestion {
  id: string;
  model_id: string;
  author: string;
  content: string;
  reply?: string;
  replied_at?: string;
  created_at: string;
}

export interface InvestorReview {
  id: string;
  model_id: string;
  scheduled_at: string;
  attendees: string[];
  notes: string;
  status: 'scheduled' | 'completed' | 'cancelled';
  created_at: string;
}
