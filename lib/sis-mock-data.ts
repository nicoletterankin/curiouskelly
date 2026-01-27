import { Learner, MessageTemplate, DashboardStats } from './sis-types';

const FIRST_NAMES = ['Emma', 'Liam', 'Olivia', 'Noah', 'Ava', 'Ethan', 'Sophia', 'Mason', 'Isabella', 'James', 'Mia', 'Lucas', 'Charlotte', 'Benjamin', 'Amelia'];
const LAST_NAMES = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Martinez', 'Wilson'];

const LOCATIONS = [
  { city: 'Irvine', region: 'California', country: 'US', flag: '🇺🇸', tz: 'America/Los_Angeles' },
  { city: 'New York', region: 'New York', country: 'US', flag: '🇺🇸', tz: 'America/New_York' },
  { city: 'London', region: 'England', country: 'GB', flag: '🇬🇧', tz: 'Europe/London' },
  { city: 'Toronto', region: 'Ontario', country: 'CA', flag: '🇨🇦', tz: 'America/Toronto' },
  { city: 'Sydney', region: 'NSW', country: 'AU', flag: '🇦🇺', tz: 'Australia/Sydney' },
  { city: 'Berlin', region: 'Berlin', country: 'DE', flag: '🇩🇪', tz: 'Europe/Berlin' },
  { city: 'Mumbai', region: 'Maharashtra', country: 'IN', flag: '🇮🇳', tz: 'Asia/Kolkata' },
  { city: 'Tokyo', region: 'Tokyo', country: 'JP', flag: '🇯🇵', tz: 'Asia/Tokyo' },
  { city: 'Paris', region: 'Île-de-France', country: 'FR', flag: '🇫🇷', tz: 'Europe/Paris' },
  { city: 'São Paulo', region: 'São Paulo', country: 'BR', flag: '🇧🇷', tz: 'America/Sao_Paulo' },
];

const LESSON_TOPICS = ['Photosynthesis', 'Gravity', 'Emotions', 'The Internet', 'Dreams', 'Volcanoes', 'Music', 'Dinosaurs', 'The Moon', 'Friendship'];

export function generateMockLearners(count: number = 50): Learner[] {
  return Array.from({ length: count }, (_, i) => {
    const hasName = Math.random() > 0.25;
    const firstName = FIRST_NAMES[Math.floor(Math.random() * FIRST_NAMES.length)];
    const lastName = LAST_NAMES[Math.floor(Math.random() * LAST_NAMES.length)];
    const location = LOCATIONS[Math.floor(Math.random() * LOCATIONS.length)];
    
    const statusRoll = Math.random();
    const subscriptionStatus: Learner['subscriptionStatus'] = 
      statusRoll < 0.5 ? 'free' :
      statusRoll < 0.65 ? 'trial' :
      statusRoll < 0.85 ? 'active' :
      statusRoll < 0.95 ? 'canceled' : 'past_due';
    
    const lessonsCompleted = Math.floor(Math.random() * 120);
    const hasStreak = lessonsCompleted > 0 && Math.random() > 0.3;
    
    const lastSeenMinutes = Math.random() < 0.15 ? Math.floor(Math.random() * 5) : Math.floor(Math.random() * 10000);
    const lastSeenAt = new Date(Date.now() - lastSeenMinutes * 60 * 1000).toISOString();
    
    return {
      id: `usr_${Math.random().toString(36).substr(2, 12)}`,
      name: hasName ? `${firstName} ${lastName}` : null,
      email: hasName ? `${firstName.toLowerCase()}.${lastName.toLowerCase()}${i}@gmail.com` : null,
      avatar: hasName ? `https://api.dicebear.com/7.x/avataaars/svg?seed=${firstName}${i}` : null,
      
      authProvider: ['google', 'apple', 'email'][Math.floor(Math.random() * 3)] as Learner['authProvider'],
      createdAt: new Date(Date.now() - Math.random() * 90 * 24 * 60 * 60 * 1000).toISOString(),
      lastSeenAt,
      
      ip: `${Math.floor(Math.random() * 200) + 10}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.xxx`,
      ipFull: `${Math.floor(Math.random() * 200) + 10}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
      city: location.city,
      region: location.region,
      country: location.country,
      countryFlag: location.flag,
      timezone: location.tz,
      
      device: ['Desktop', 'Mobile', 'Tablet'][Math.floor(Math.random() * 3)] as Learner['device'],
      browser: ['Chrome', 'Safari', 'Firefox', 'Edge'][Math.floor(Math.random() * 4)],
      os: ['Windows', 'macOS', 'iOS', 'Android'][Math.floor(Math.random() * 4)],
      
      subscriptionStatus,
      subscriptionPlan: subscriptionStatus === 'active' ? ['monthly', 'yearly', 'lifetime'][Math.floor(Math.random() * 3)] as Learner['subscriptionPlan'] : null,
      stripeCustomerId: subscriptionStatus !== 'free' ? `cus_${Math.random().toString(36).substr(2, 14)}` : null,
      trialEndsAt: subscriptionStatus === 'trial' ? new Date(Date.now() + (Math.random() * 14) * 24 * 60 * 60 * 1000).toISOString() : null,
      currentPeriodEnd: subscriptionStatus === 'active' ? new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString() : null,
      
      currentDay: Math.min(lessonsCompleted + 1, 365),
      lessonsCompleted,
      totalTimeSpent: lessonsCompleted * (120 + Math.floor(Math.random() * 120)),
      currentStreak: hasStreak ? Math.floor(Math.random() * 30) + 1 : 0,
      longestStreak: Math.floor(Math.random() * 50),
      lastLessonTopic: lessonsCompleted > 0 ? LESSON_TOPICS[Math.floor(Math.random() * LESSON_TOPICS.length)] : null,
      kellyAge: [5, 12, 18, 35, 55, 77][Math.floor(Math.random() * 6)] as Learner['kellyAge'],
      preferredLanguage: 'en',
      
      emailOptIn: Math.random() > 0.2,
      pushOptIn: Math.random() > 0.4,
      emailsSent: Math.floor(Math.random() * 10),
      emailsOpened: Math.floor(Math.random() * 8),
      
      isOnline: lastSeenMinutes < 5,
      isVIP: Math.random() > 0.95,
      needsAttention: Math.random() > 0.9,
    };
  });
}

export const MESSAGE_TEMPLATES: MessageTemplate[] = [
  {
    id: 'welcome',
    name: 'Welcome',
    type: 'email',
    subject: 'Welcome to Curious Kelly!',
    body: 'Hi {{name}},\n\nWelcome to Curious Kelly! Kelly is excited to teach you something new every day.\n\nYour first lesson is ready. See you there!\n\n— The Curious Kelly Team',
    variables: ['name'],
  },
  {
    id: 'streak-reminder',
    name: 'Streak Reminder',
    type: 'email',
    subject: "Don't break your {{streak}}-day streak!",
    body: "Hi {{name}},\n\nYou've got a {{streak}}-day streak going! Don't let it slip.\n\nDay {{day}} is waiting for you.\n\nSee you soon,\nKelly",
    variables: ['name', 'streak', 'day'],
  },
  {
    id: 'trial-ending',
    name: 'Trial Ending',
    type: 'email',
    subject: 'Your trial ends soon',
    body: "Hi {{name}},\n\nYour Curious Kelly trial is ending soon. You've completed {{lessons}} lessons so far!\n\nSubscribe now to keep learning with Kelly every day.\n\n— The Curious Kelly Team",
    variables: ['name', 'lessons'],
  },
  {
    id: 're-engagement',
    name: 'Re-engagement',
    type: 'email',
    subject: 'Kelly misses you!',
    body: "Hi {{name}},\n\nIt's been a while since your last lesson. Kelly has new things to teach you!\n\nCome back and continue your journey on Day {{day}}.\n\n— The Curious Kelly Team",
    variables: ['name', 'day'],
  },
  {
    id: 'daily-push',
    name: 'Daily Lesson (Push)',
    type: 'push',
    subject: "Day {{day}} is ready!",
    body: "{{name}}, your daily lesson with Kelly is waiting.",
    variables: ['name', 'day'],
  },
];

export function calculateStats(learners: Learner[]): DashboardStats {
  return {
    total: learners.length,
    online: learners.filter(l => l.isOnline).length,
    subscribed: learners.filter(l => l.subscriptionStatus === 'active').length,
    trial: learners.filter(l => l.subscriptionStatus === 'trial').length,
    free: learners.filter(l => l.subscriptionStatus === 'free').length,
    activeStreaks: learners.filter(l => l.currentStreak > 0).length,
  };
}
