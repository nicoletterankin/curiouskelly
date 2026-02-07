import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * KELLY REACTIONS API
 * Returns contextual messages for system events (welcome, loading, support, etc.)
 * 
 * GET /api/kelly/reactions?event=welcome&age=adult&language=en
 */

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const event = searchParams.get('event') || 'welcome'
  const age = searchParams.get('age') || 'all'
  const language = searchParams.get('language') || 'en'

  try {
    // Query with fallback: try specific age, then 'all'
    const reactions = await sql`
      SELECT script_text, audio_url, video_url, visual_url, age_group
      FROM kelly_reactions
      WHERE trigger_event = ${event}
        AND language = ${language}
        AND (age_group = ${age} OR age_group = 'all')
        AND is_active = true
      ORDER BY 
        CASE WHEN age_group = ${age} THEN 0 ELSE 1 END,
        priority DESC
      LIMIT 1
    `

    if (reactions && reactions.length > 0) {
      const reaction = reactions[0]
      return NextResponse.json({
        script: reaction.script_text,
        audioUrl: reaction.audio_url,
        videoUrl: reaction.video_url,
        visualUrl: reaction.visual_url,
        event,
        ageGroup: reaction.age_group,
      })
    }

    // Fallback messages if nothing in DB
    const fallbacks: Record<string, string> = {
      welcome: "Hi! I'm Kelly, your daily learning companion. Ready to explore something amazing?",
      loading: "Pardon me while my systems come online... gathering today's wonders!",
      support: "I'm here to help! What's on your mind?",
      daily_greeting: "Great to see you! Ready for today's lesson?",
      encouragement: "You're doing great! Keep exploring!",
      goodbye: "Thanks for learning with me! See you tomorrow!",
    }

    return NextResponse.json({
      script: fallbacks[event] || fallbacks.welcome,
      audioUrl: null,
      videoUrl: null,
      visualUrl: null,
      event,
      ageGroup: 'all',
      source: 'fallback',
    })

  } catch (error) {
    console.error('[kelly/reactions] Error:', error)
    return NextResponse.json({
      script: "Pardon me while my systems come online... I'll be right with you!",
      audioUrl: null,
      videoUrl: null,
      visualUrl: null,
      event,
      source: 'error_fallback',
    })
  }
}
