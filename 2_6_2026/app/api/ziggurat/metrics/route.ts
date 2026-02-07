import { NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * Real-time Metrics API for Ziggurat Financial Model
 * 
 * Provides actual data from the database to validate investor projections:
 * - Learner counts (total, active, by tier)
 * - Revenue metrics (MRR, ARR, ARPU)
 * - Engagement metrics (DAU, retention, completion rates)
 * - Content metrics (lessons, assets)
 */

export async function GET() {
  try {
    // Get learner metrics
    const learnerMetrics = await sql`
      SELECT 
        COUNT(*)::int as total_learners,
        COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days')::int as new_learners_30d,
        COUNT(*) FILTER (WHERE updated_at > NOW() - INTERVAL '7 days')::int as active_7d,
        COUNT(*) FILTER (WHERE updated_at > NOW() - INTERVAL '1 day')::int as active_24h
      FROM learners
    `.catch(() => [{ total_learners: 0, new_learners_30d: 0, active_7d: 0, active_24h: 0 }])

    // Get subscriber metrics  
    const subscriberMetrics = await sql`
      SELECT
        COUNT(*)::int as total_subscribers,
        COUNT(*) FILTER (WHERE plan = 'monthly')::int as monthly_subs,
        COUNT(*) FILTER (WHERE plan = 'annual')::int as annual_subs,
        COUNT(*) FILTER (WHERE plan = 'lifetime')::int as lifetime_subs,
        COUNT(*) FILTER (WHERE plan = 'family')::int as family_subs
      FROM subscribers
      WHERE status = 'active'
    `.catch(() => [{ total_subscribers: 0, monthly_subs: 0, annual_subs: 0, lifetime_subs: 0, family_subs: 0 }])

    // Get lesson completion metrics
    const engagementMetrics = await sql`
      SELECT
        COUNT(*)::int as total_completions,
        COUNT(DISTINCT learner_id)::int as unique_learners_completed,
        ROUND(AVG(completion_percent)::numeric, 1)::float as avg_completion_rate
      FROM lesson_progress
    `.catch(() => [{ total_completions: 0, unique_learners_completed: 0, avg_completion_rate: 0 }])

    // Get content metrics
    const contentMetrics = await sql`
      SELECT 
        (SELECT COUNT(*)::int FROM lessons) as total_lessons,
        (SELECT COUNT(*)::int FROM kelly_lesson_assets WHERE status = 'complete') as complete_assets,
        (SELECT COUNT(DISTINCT day_number) FROM kelly_lesson_assets WHERE audio_url IS NOT NULL) as days_with_audio
    `.catch(() => [{ total_lessons: 365, complete_assets: 0, days_with_audio: 0 }])

    // Get investor model metrics
    const investorMetrics = await sql`
      SELECT
        COUNT(*)::int as total_models,
        SUM(view_count)::int as total_views,
        (SELECT COUNT(*)::int FROM investor_questions) as total_questions,
        (SELECT COUNT(*)::int FROM investor_reviews WHERE status = 'scheduled') as upcoming_reviews
      FROM investor_models
    `.catch(() => [{ total_models: 0, total_views: 0, total_questions: 0, upcoming_reviews: 0 }])

    // Calculate revenue metrics
    const subs = subscriberMetrics[0] || {}
    const monthlyRevenue = (subs.monthly_subs || 0) * 9 // $9/mo
    const annualMonthlyEquiv = (subs.annual_subs || 0) * 7.5 // $90/yr = $7.50/mo
    const lifetimeMonthlyEquiv = (subs.lifetime_subs || 0) * 3.75 // $180 / 48 mo
    const familyMonthlyEquiv = (subs.family_subs || 0) * 15 // $180/yr = $15/mo
    
    const mrr = monthlyRevenue + annualMonthlyEquiv + lifetimeMonthlyEquiv + familyMonthlyEquiv
    const arr = mrr * 12

    // Build response
    const metrics = {
      timestamp: new Date().toISOString(),
      
      learners: {
        total: learnerMetrics[0]?.total_learners || 0,
        new30d: learnerMetrics[0]?.new_learners_30d || 0,
        active7d: learnerMetrics[0]?.active_7d || 0,
        active24h: learnerMetrics[0]?.active_24h || 0,
      },
      
      subscribers: {
        total: subs.total_subscribers || 0,
        monthly: subs.monthly_subs || 0,
        annual: subs.annual_subs || 0,
        lifetime: subs.lifetime_subs || 0,
        family: subs.family_subs || 0,
      },
      
      revenue: {
        mrr: Math.round(mrr),
        arr: Math.round(arr),
        arpu: subs.total_subscribers > 0 ? Math.round(mrr / subs.total_subscribers * 100) / 100 : 0,
      },
      
      engagement: {
        totalCompletions: engagementMetrics[0]?.total_completions || 0,
        uniqueLearners: engagementMetrics[0]?.unique_learners_completed || 0,
        avgCompletionRate: engagementMetrics[0]?.avg_completion_rate || 0,
      },
      
      content: {
        totalLessons: contentMetrics[0]?.total_lessons || 365,
        completeAssets: contentMetrics[0]?.complete_assets || 0,
        daysWithAudio: contentMetrics[0]?.days_with_audio || 0,
      },
      
      investors: {
        totalModels: investorMetrics[0]?.total_models || 0,
        totalViews: investorMetrics[0]?.total_views || 0,
        totalQuestions: investorMetrics[0]?.total_questions || 0,
        upcomingReviews: investorMetrics[0]?.upcoming_reviews || 0,
      },
      
      // Key ratios for investor analysis
      ratios: {
        conversionRate: learnerMetrics[0]?.total_learners > 0 
          ? Math.round((subs.total_subscribers || 0) / learnerMetrics[0].total_learners * 10000) / 100 
          : 0,
        retentionRate7d: learnerMetrics[0]?.total_learners > 0
          ? Math.round((learnerMetrics[0]?.active_7d || 0) / learnerMetrics[0].total_learners * 10000) / 100
          : 0,
        contentCompletionRate: contentMetrics[0]?.total_lessons > 0
          ? Math.round((contentMetrics[0]?.days_with_audio || 0) / contentMetrics[0].total_lessons * 10000) / 100
          : 0,
      }
    }

    return NextResponse.json(metrics)
    
  } catch (error) {
    console.error('Error fetching metrics:', error)
    return NextResponse.json({ 
      error: 'Failed to fetch metrics',
      timestamp: new Date().toISOString(),
      // Return zeros on error
      learners: { total: 0, new30d: 0, active7d: 0, active24h: 0 },
      subscribers: { total: 0, monthly: 0, annual: 0, lifetime: 0, family: 0 },
      revenue: { mrr: 0, arr: 0, arpu: 0 },
      engagement: { totalCompletions: 0, uniqueLearners: 0, avgCompletionRate: 0 },
      content: { totalLessons: 365, completeAssets: 0, daysWithAudio: 0 },
      investors: { totalModels: 0, totalViews: 0, totalQuestions: 0, upcomingReviews: 0 },
      ratios: { conversionRate: 0, retentionRate7d: 0, contentCompletionRate: 0 }
    }, { status: 200 }) // Return 200 even on error to not break UI
  }
}
