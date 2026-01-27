import { Learner } from './sis-types';

export function formatTimeAgo(date: string | null | undefined): string {
  if (!date) return 'Never';
  
  const now = Date.now();
  const then = new Date(date).getTime();
  const diff = now - then;
  
  if (diff < 0 || diff < 60 * 1000) return 'Just now';
  if (diff < 60 * 60 * 1000) return `${Math.floor(diff / (60 * 1000))}m ago`;
  if (diff < 24 * 60 * 60 * 1000) return `${Math.floor(diff / (60 * 60 * 1000))}h ago`;
  return `${Math.floor(diff / (24 * 60 * 60 * 1000))}d ago`;
}

export function formatDuration(seconds: number): string {
  if (!seconds || seconds <= 0) return '—';
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

export function anonymizeIP(ip: string | null): string {
  if (!ip) return '—';
  if (ip.includes(':')) {
    // IPv6
    const parts = ip.split(':');
    parts[parts.length - 1] = 'xxxx';
    return parts.join(':');
  }
  // IPv4
  return ip.replace(/\.\d+$/, '.xxx');
}

export function getSubscriptionBadge(status: Learner['subscriptionStatus'], plan?: Learner['subscriptionPlan']): { label: string; color: string } {
  switch (status) {
    case 'free':
      return { label: 'Free', color: 'bg-gray-500/20 text-gray-400' };
    case 'trial':
      return { label: 'Trial', color: 'bg-yellow-500/20 text-yellow-400' };
    case 'active':
      return { label: plan || 'Active', color: 'bg-green-500/20 text-green-400' };
    case 'canceled':
      return { label: 'Canceled', color: 'bg-red-500/20 text-red-400' };
    case 'past_due':
      return { label: 'Past Due', color: 'bg-orange-500/20 text-orange-400' };
    default:
      return { label: 'Unknown', color: 'bg-gray-500/20 text-gray-400' };
  }
}

export function personalizeMessage(template: string, learner: Learner): string {
  const vars: Record<string, string> = {
    name: learner.name || 'Learner',
    email: learner.email || '',
    streak: String(learner.currentStreak || 0),
    lessons: String(learner.lessonsCompleted || 0),
    day: String(learner.currentDay || 1),
  };
  
  return template.replace(/\{\{(\w+)\}\}/g, (match, key) => {
    return vars[key] ?? match;
  });
}

export function filterLearners(
  learners: Learner[],
  filters: {
    search?: string;
    status?: string;
    country?: string;
  }
): Learner[] {
  let result = [...learners];
  
  if (filters.search) {
    const q = filters.search.toLowerCase().trim();
    result = result.filter(l =>
      l.name?.toLowerCase().includes(q) ||
      l.email?.toLowerCase().includes(q) ||
      l.id.toLowerCase().includes(q) ||
      l.city?.toLowerCase().includes(q) ||
      l.country?.toLowerCase().includes(q)
    );
  }
  
  if (filters.status && filters.status !== 'all') {
    result = result.filter(l => l.subscriptionStatus === filters.status);
  }
  
  if (filters.country && filters.country !== 'all') {
    result = result.filter(l => l.country === filters.country);
  }
  
  return result;
}

export function getUniqueCountries(learners: Learner[]): { code: string; flag: string }[] {
  const map = new Map<string, string>();
  learners.forEach(l => {
    if (l.country && l.countryFlag) {
      map.set(l.country, l.countryFlag);
    }
  });
  return Array.from(map.entries()).map(([code, flag]) => ({ code, flag })).sort((a, b) => a.code.localeCompare(b.code));
}
