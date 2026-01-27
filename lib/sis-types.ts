export interface Learner {
  // Identity
  id: string;
  name: string | null;
  email: string | null;
  avatar: string | null;
  
  // Auth
  authProvider: 'google' | 'apple' | 'email';
  createdAt: string;
  lastSeenAt: string | null;
  
  // Location (from Vercel geo headers)
  ip: string | null;
  ipFull: string | null;
  city: string | null;
  region: string | null;
  country: string | null;
  countryFlag: string;
  timezone: string | null;
  
  // Device
  device: 'Desktop' | 'Mobile' | 'Tablet' | null;
  browser: string | null;
  os: string | null;
  
  // Subscription (from Stripe)
  subscriptionStatus: 'free' | 'trial' | 'active' | 'canceled' | 'past_due';
  subscriptionPlan: 'monthly' | 'yearly' | 'lifetime' | null;
  stripeCustomerId: string | null;
  trialEndsAt: string | null;
  currentPeriodEnd: string | null;
  
  // Learning Progress
  currentDay: number;
  lessonsCompleted: number;
  totalTimeSpent: number;
  currentStreak: number;
  longestStreak: number;
  lastLessonTopic: string | null;
  kellyAge: 5 | 12 | 18 | 35 | 55 | 77;
  preferredLanguage: string;
  
  // Communication
  emailOptIn: boolean;
  pushOptIn: boolean;
  emailsSent: number;
  emailsOpened: number;
  
  // Flags
  isOnline: boolean;
  isVIP: boolean;
  needsAttention: boolean;
}

export interface MessageTemplate {
  id: string;
  name: string;
  type: 'email' | 'push';
  subject: string;
  body: string;
  variables: string[];
}

export interface DashboardStats {
  total: number;
  online: number;
  subscribed: number;
  trial: number;
  free: number;
  activeStreaks: number;
}
