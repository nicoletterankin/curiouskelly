// ============================================
// CURIOUS KELLY - API CLIENT
// ============================================

import { supabase } from './auth.js';

// ============================================
// LESSON API
// ============================================

function getTodayDayNumber() {
  if (
    typeof window !== 'undefined' &&
    window.KellyTime?.getLessonDayForTimeZone &&
    window.KellyTime?.getUserTimeZone
  ) {
    return window.KellyTime.getLessonDayForTimeZone(Date.now(), window.KellyTime.getUserTimeZone());
  }

  const now = new Date();
  const startOfYear = new Date(now.getFullYear(), 0, 0);
  const diff = now - startOfYear;
  const day = Math.floor(diff / (1000 * 60 * 60 * 24));
  return Math.max(1, Math.min(365, day || 1));
}

export async function getTodaysLesson() {
  let currentDay = getTodayDayNumber();

  try {
    const {
      data: { user },
    } = await supabase.auth.getUser();
    if (user) {
      const { data: userData } = await supabase
        .from('users')
        .select('current_day')
        .eq('id', user.id)
        .single();
      if (userData?.current_day) currentDay = userData.current_day;
    }
  } catch {
    // Guest mode: ignore
  }

  const { data: lesson, error } = await supabase
    .from('core_lessons')
    .select('*')
    .eq('day_number', currentDay)
    .single();

  if (error) throw error;

  return { lesson, day: currentDay };
}

export async function getLesson(lessonId) {
  const { data: lesson, error } = await supabase
    .from('core_lessons')
    .select('*')
    .eq('id', lessonId)
    .single();

  if (error) throw error;

  return lesson;
}

export async function getLessonByDay(dayNumber) {
  const { data: lesson, error } = await supabase
    .from('core_lessons')
    .select('*')
    .eq('day_number', dayNumber)
    .single();

  if (error) throw error;

  return lesson;
}

export async function getCalendar() {
  const { data, error } = await supabase
    .from('core_lessons')
    .select('id, day_number, topic, category, tags, duration_min, duration_max')
    .order('day_number', { ascending: true });

  if (error) throw error;

  return (data || []).map(l => ({
    id: l.id,
    day_number: l.day_number,
    title: l.topic,
    subtitle: l.category || '',
    duration_seconds: (l.duration_max || l.duration_min || 8) * 60,
    difficulty: null,
    tags: l.tags || [],
  }));
}

// ============================================
// PROGRESS API
// ============================================

export async function getUserProgress() {
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) throw new Error('Not authenticated');

  const { data: progress, error } = await supabase
    .from('user_progress')
    .select(
      `
      *,
      lessons:lesson_id (
        day_number,
        title,
        duration_seconds
      )
    `
    )
    .eq('user_id', user.id)
    .order('started_at', { ascending: false });

  if (error) throw error;

  return progress;
}

export async function getLessonProgress(lessonId) {
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) throw new Error('Not authenticated');

  const { data: progress, error } = await supabase
    .from('user_progress')
    .select('*')
    .eq('user_id', user.id)
    .eq('lesson_id', lessonId)
    .single();

  if (error && error.code !== 'PGRST116') throw error; // PGRST116 = not found, which is ok

  return progress;
}

export async function updateProgress(lessonId, progressData) {
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) throw new Error('Not authenticated');

  const { data, error } = await supabase
    .from('user_progress')
    .upsert(
      {
        user_id: user.id,
        lesson_id: lessonId,
        ...progressData,
        updated_at: new Date().toISOString(),
      },
      {
        onConflict: 'user_id,lesson_id',
      }
    )
    .select()
    .single();

  if (error) throw error;

  return data;
}

export async function completeLesson(lessonId) {
  return updateProgress(lessonId, {
    completed: true,
    progress_percent: 100,
    completed_at: new Date().toISOString(),
  });
}

// ============================================
// USER API
// ============================================

export async function getUserProfile() {
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) throw new Error('Not authenticated');

  const { data: profile, error } = await supabase
    .from('users')
    .select('*')
    .eq('id', user.id)
    .single();

  if (error) throw error;

  return profile;
}

export async function updateUserProfile(updates) {
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) throw new Error('Not authenticated');

  const { data, error } = await supabase
    .from('users')
    .update(updates)
    .eq('id', user.id)
    .select()
    .single();

  if (error) throw error;

  return data;
}

export async function getUserStreak() {
  const profile = await getUserProfile();
  return {
    streak_days: profile.streak_days || 0,
    current_day: profile.current_day || 1,
    last_lesson_at: profile.last_lesson_at,
  };
}

// ============================================
// AFFILIATE API
// ============================================

export async function submitAffiliateApplication(applicationData) {
  const { data, error } = await supabase
    .from('affiliate_applications')
    .insert(applicationData)
    .select()
    .single();

  if (error) throw error;

  return data;
}

export async function getAffiliateData() {
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) throw new Error('Not authenticated');

  const { data: affiliate, error } = await supabase
    .from('affiliates')
    .select('*')
    .eq('user_id', user.id)
    .single();

  if (error && error.code !== 'PGRST116') throw error;

  return affiliate;
}

export async function getAffiliateReferrals() {
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) throw new Error('Not authenticated');

  const { data: affiliate } = await supabase
    .from('affiliates')
    .select('id')
    .eq('user_id', user.id)
    .single();

  if (!affiliate) return [];

  const { data: referrals, error } = await supabase
    .from('referrals')
    .select(
      `
      *,
      referred_user:referred_user_id (
        email,
        name,
        created_at
      )
    `
    )
    .eq('affiliate_id', affiliate.id)
    .order('created_at', { ascending: false });

  if (error) throw error;

  return referrals;
}

// ============================================
// ENTERPRISE API
// ============================================

export async function submitEnterpriseInquiry(inquiryData) {
  const { data, error } = await supabase
    .from('enterprise_inquiries')
    .insert(inquiryData)
    .select()
    .single();

  if (error) throw error;

  return data;
}

// ============================================
// NEWSLETTER API
// ============================================

export async function subscribeToNewsletter(email) {
  const { data, error } = await supabase
    .from('newsletter_subscribers')
    .insert({ email, source: 'website' })
    .select()
    .single();

  if (error) {
    if (error.code === '23505') {
      // Duplicate
      throw new Error('Already subscribed');
    }
    throw error;
  }

  return data;
}

// ============================================
// ANALYTICS API
// ============================================

export async function logEvent(eventType, eventData = {}) {
  const {
    data: { user },
  } = await supabase.auth.getUser();

  await supabase.from('analytics_events').insert({
    user_id: user?.id || null,
    event_type: eventType,
    event_data: eventData,
    session_id: sessionStorage.getItem('session_id'),
    user_agent: navigator.userAgent,
  });
}
