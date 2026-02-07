import { NextResponse } from 'next/server'
import { sql } from '@/lib/db'

export async function GET() {
  try {
    // Get stats from database
    const [todayStats, deviceStats, referrerStats, recentVisitors] = await Promise.all([
      // Today's visits
      sql`
        SELECT 
          COUNT(DISTINCT session_id) as total_sessions,
          AVG((event_data->>'timeOnPage')::int) FILTER (WHERE event_type = 'session_end') as avg_time,
          AVG((event_data->>'scrollDepth')::int) FILTER (WHERE event_type = 'session_end') as avg_scroll
        FROM analytics_events
        WHERE created_at > NOW() - INTERVAL '24 hours'
      `.catch(() => [{ total_sessions: 0, avg_time: 0, avg_scroll: 0 }]),
      
      // Device breakdown
      sql`
        SELECT 
          event_data->'device'->>'type' as device_type,
          COUNT(DISTINCT session_id) as count
        FROM analytics_events
        WHERE event_type = 'pageview' AND created_at > NOW() - INTERVAL '24 hours'
        GROUP BY device_type
      `.catch(() => []),
      
      // Top referrers
      sql`
        SELECT 
          COALESCE(NULLIF(event_data->>'referrer', ''), 'Direct') as source,
          COUNT(DISTINCT session_id) as count
        FROM analytics_events
        WHERE event_type = 'pageview' AND created_at > NOW() - INTERVAL '24 hours'
        GROUP BY source
        ORDER BY count DESC
        LIMIT 5
      `.catch(() => []),
      
      // Recent visitors
      sql`
        SELECT DISTINCT ON (session_id)
          session_id as id,
          created_at as timestamp,
          ip_address as ip,
          city, region, country,
          event_data->'device' as device,
          event_data->>'referrer' as referrer,
          (SELECT (event_data->>'timeOnPage')::int FROM analytics_events e2 
           WHERE e2.session_id = analytics_events.session_id AND e2.event_type = 'session_end' LIMIT 1) as time_on_page,
          (SELECT (event_data->>'scrollDepth')::int FROM analytics_events e2 
           WHERE e2.session_id = analytics_events.session_id AND e2.event_type = 'session_end' LIMIT 1) as scroll_depth
        FROM analytics_events
        WHERE event_type = 'pageview' AND created_at > NOW() - INTERVAL '1 hour'
        ORDER BY session_id, created_at DESC
        LIMIT 20
      `.catch(() => []),
    ])
    
    // Process device breakdown
    const deviceBreakdown = { desktop: 0, mobile: 0, tablet: 0 }
    const totalDevices = deviceStats.reduce((sum: number, d: { count: number }) => sum + Number(d.count), 0) || 1
    for (const d of deviceStats) {
      const type = (d.device_type || '').toLowerCase()
      if (type === 'desktop') deviceBreakdown.desktop = Math.round((Number(d.count) / totalDevices) * 100)
      else if (type === 'mobile') deviceBreakdown.mobile = Math.round((Number(d.count) / totalDevices) * 100)
      else if (type === 'tablet') deviceBreakdown.tablet = Math.round((Number(d.count) / totalDevices) * 100)
    }
    
    // Count live visitors (active in last 5 minutes)
    const liveResult = await sql`
      SELECT COUNT(DISTINCT session_id) as count
      FROM analytics_events
      WHERE created_at > NOW() - INTERVAL '5 minutes'
    `.catch(() => [{ count: 0 }])
    
    return NextResponse.json({
      liveVisitors: Number(liveResult[0]?.count) || 0,
      todayVisits: Number(todayStats[0]?.total_sessions) || 0,
      avgTimeOnPage: Math.round(Number(todayStats[0]?.avg_time) || 0),
      avgScrollDepth: Math.round(Number(todayStats[0]?.avg_scroll) || 0),
      deviceBreakdown,
      topReferrers: referrerStats.map((r: { source: string; count: number }) => ({
        source: r.source,
        count: Number(r.count),
      })),
      recentVisitors: recentVisitors.map((v: Record<string, unknown>) => ({
        id: v.id,
        timestamp: v.timestamp,
        ip: v.ip,
        location: { city: v.city, region: v.region, country: v.country },
        device: v.device || { type: 'Unknown', os: 'Unknown', browser: 'Unknown' },
        referrer: v.referrer || '',
        timeOnPage: v.time_on_page || 0,
        scrollDepth: v.scroll_depth || 0,
        interactions: [],
      })),
    })
  } catch (error) {
    console.error('[analytics/stats] Error:', error)
    
    // Return empty stats on error
    return NextResponse.json({
      liveVisitors: 0,
      todayVisits: 0,
      avgTimeOnPage: 0,
      avgScrollDepth: 0,
      deviceBreakdown: { desktop: 0, mobile: 0, tablet: 0 },
      topReferrers: [],
      recentVisitors: [],
    })
  }
}
