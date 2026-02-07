import { neon, NeonQueryFunction } from "@neondatabase/serverless"
import { NextResponse } from "next/server"

// Lazy-initialized SQL client - prevents build-time connection errors
let _sql: NeonQueryFunction<false, false> | null = null
function getSql() {
  if (!_sql) {
    _sql = neon(process.env.DATABASE_URL || process.env.NEON_DATABASE_URL || "")
  }
  return _sql
}

interface EvalResult {
  test: string
  passed: boolean
  message: string
  data?: unknown
  category: string
}

export async function GET() {
  const results: EvalResult[] = []
  const startTime = Date.now()

  function log(category: string, test: string, passed: boolean, message: string, data?: unknown) {
    results.push({ category, test, passed, message, data })
  }

  try {
    // ============ DATABASE ============
    const sql = getSql()
    const dbCheck = await sql`SELECT NOW() as time, current_database() as db`
    log("database", "Connection", true, `Connected to ${dbCheck[0]?.db}`)

    // ============ LESSONS ============
    const lessonCount = await sql`SELECT COUNT(DISTINCT day_of_year) as count FROM lessons`
    const lCount = Number(lessonCount[0]?.count || 0)
    log("lessons", "Coverage (365 days)", lCount >= 365, `${lCount}/365 days`)

    const today = await sql`SELECT title, topic FROM lessons WHERE day_of_year = 24 LIMIT 1`
    log("lessons", "Day 24 (Today)", today.length > 0, today[0]?.title || "Missing")

    const missingPhases = await sql`
      SELECT COUNT(*) as count FROM lessons 
      WHERE hook_script IS NULL OR story_script IS NULL OR wonder_script IS NULL 
            OR action_script IS NULL OR wisdom_script IS NULL
    `
    log("lessons", "Phase Completeness", Number(missingPhases[0]?.count) === 0, 
        `${missingPhases[0]?.count} incomplete lessons`)

    // ============ KELLY ASSETS ============
    const assetStats = await sql`
      SELECT COUNT(DISTINCT day_number) as days,
             COUNT(*) as total,
             COUNT(DISTINCT age_group) as ages,
             COUNT(DISTINCT language) as langs,
             COUNT(*) FILTER (WHERE video_url IS NOT NULL) as videos,
             COUNT(*) FILTER (WHERE audio_url IS NOT NULL) as audios
      FROM kelly_lesson_assets
    `
    const as = assetStats[0]
    log("assets", "Kelly Lesson Assets", Number(as?.days) > 0, 
        `${as?.days} days, ${as?.videos} videos, ${as?.audios} audios`)
    log("assets", "Age Groups", Number(as?.ages) >= 3, `${as?.ages} age groups`)
    log("assets", "Languages", Number(as?.langs) > 0, `${as?.langs} languages`)

    // Phase-specific audio
    const phaseAudio = await sql`
      SELECT 
        COUNT(*) FILTER (WHERE hook_audio IS NOT NULL) as hook,
        COUNT(*) FILTER (WHERE story_audio IS NOT NULL) as story,
        COUNT(*) FILTER (WHERE wonder_audio IS NOT NULL) as wonder,
        COUNT(*) FILTER (WHERE action_audio IS NOT NULL) as action,
        COUNT(*) FILTER (WHERE wisdom_audio IS NOT NULL) as wisdom
      FROM kelly_lesson_assets
    `
    const pa = phaseAudio[0]
    const hasPhaseAudio = Number(pa?.hook) > 0 || Number(pa?.story) > 0
    log("assets", "Phase Audio", hasPhaseAudio, 
        `H:${pa?.hook} S:${pa?.story} W:${pa?.wonder} A:${pa?.action} W:${pa?.wisdom}`)

    // ============ KELLY REACTIONS ============
    const reactions = await sql`
      SELECT trigger_event, COUNT(*) as count FROM kelly_reactions GROUP BY trigger_event
    `
    const reactionEvents = reactions.map(r => r.trigger_event)
    const requiredReactions = ["welcome", "support", "daily_greeting"]
    for (const evt of requiredReactions) {
      log("reactions", `Reaction: ${evt}`, reactionEvents.includes(evt), 
          reactionEvents.includes(evt) ? "Available" : "MISSING")
    }

    // ============ LANGUAGES ============
    const languages = await sql`
      SELECT COUNT(*) as count FROM languages WHERE status = 'active'
    `
    log("i18n", "Active Languages", Number(languages[0]?.count) > 0, 
        `${languages[0]?.count} active`)

    const englishCheck = await sql`SELECT * FROM languages WHERE language_code = 'en' LIMIT 1`
    log("i18n", "English Language", englishCheck.length > 0, 
        englishCheck.length > 0 ? "Active" : "MISSING")

    // ============ TRANSLATIONS ============
    const translations = await sql`
      SELECT COUNT(DISTINCT language) as langs, COUNT(*) as total FROM active_translations
    `
    log("i18n", "Active Translations", Number(translations[0]?.langs) > 0, 
        `${translations[0]?.langs} languages, ${translations[0]?.total} total`)

    // ============ VIDEO QUEUE ============
    const videoQueue = await sql`
      SELECT status, COUNT(*) as count FROM video_jobs GROUP BY status
    `
    for (const row of videoQueue) {
      log("queue", `Video Jobs (${row.status})`, true, `${row.count}`)
    }

    const stuckJobs = await sql`
      SELECT COUNT(*) as count FROM video_jobs 
      WHERE status = 'processing' AND created_at < NOW() - INTERVAL '1 hour'
    `
    log("queue", "Stuck Jobs (>1hr)", Number(stuckJobs[0]?.count) === 0, 
        `${stuckJobs[0]?.count} stuck`)

    // ============ ARCHETYPES & TONES ============
    const archetypes = await sql`
      SELECT DISTINCT archetype FROM kelly_lesson_assets WHERE archetype IS NOT NULL
    `
    log("archetypes", "Available Archetypes", archetypes.length > 0, 
        `${archetypes.length}: ${archetypes.slice(0, 5).map(a => a.archetype).join(", ")}...`)

    const tones = await sql`SELECT COUNT(*) as count FROM tone_profiles`
    log("archetypes", "Tone Profiles", Number(tones[0]?.count) > 0, 
        `${tones[0]?.count} profiles`)

    // ============ VISUAL AIDS ============
    const visuals = await sql`
      SELECT COUNT(*) as count FROM concept_assets WHERE approved = true
    `
    log("visuals", "Approved Visuals", Number(visuals[0]?.count) > 0, 
        `${visuals[0]?.count} approved`)

    // ============ USER SYSTEMS ============
    const learners = await sql`SELECT COUNT(*) as count FROM learners`
    log("users", "Learners", true, `${learners[0]?.count}`)

    const progress = await sql`SELECT COUNT(*) as count FROM user_progress`
    log("users", "Progress Records", true, `${progress[0]?.count}`)

    // ============ API SIMULATION ============
    // Note: kelly_lesson_assets only has audio_url, not video_url
    const apiVideoSim = await sql`
      SELECT audio_url, age_group FROM kelly_lesson_assets
      WHERE day_number = 24 AND language = 'en'
      ORDER BY CASE WHEN age_group = 'adult' THEN 0 ELSE 1 END
      LIMIT 1
    `
    log("api", "/api/video/url (Day 24)", apiVideoSim.length > 0, 
        apiVideoSim.length > 0 ? `Found: ${apiVideoSim[0]?.age_group}` : "No results")

    // ============ SUMMARY ============
    const passed = results.filter(r => r.passed).length
    const failed = results.filter(r => !r.passed).length
    const duration = Date.now() - startTime

    return NextResponse.json({
      success: failed === 0,
      summary: { passed, failed, total: results.length, duration_ms: duration },
      results,
      failures: results.filter(r => !r.passed),
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
      results,
    }, { status: 500 })
  }
}
