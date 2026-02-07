import { neon, neonConfig, NeonQueryFunction } from '@neondatabase/serverless'
import { NextRequest, NextResponse } from 'next/server'

neonConfig.disableWarningInBrowsers = true

// Lazy-initialized SQL client - prevents build-time connection errors
let _sql: NeonQueryFunction<false, false> | null = null
function getSql() {
  if (!_sql) {
    _sql = neon(process.env.DATABASE_URL!)
  }
  return _sql
}

// Search lessons by topic, title, or content
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const query = searchParams.get('q')
  const limit = searchParams.get('limit') || '10'
  
  if (!query || query.length < 2) {
    return NextResponse.json({ error: 'Query must be at least 2 characters' }, { status: 400 })
  }
  
  // Limit query length and sanitize special LIKE characters
  const sanitizedQuery = query
    .slice(0, 100) // Max 100 chars
    .replace(/[%_\\]/g, '') // Remove LIKE wildcards
    .trim()
  
  if (!sanitizedQuery) {
    return NextResponse.json({ error: 'Invalid search query' }, { status: 400 })
  }
  
  try {
    // Search across multiple fields using ILIKE for fuzzy matching
    const searchTerm = `%${sanitizedQuery.toLowerCase()}%`
    const sql = getSql()
    
    const results = await sql`
      SELECT DISTINCT
        day_of_year,
        topic,
        title,
        summary,
        hook_script
      FROM lessons
      WHERE 
        LOWER(topic) LIKE ${searchTerm}
        OR LOWER(title) LIKE ${searchTerm}
        OR LOWER(summary) LIKE ${searchTerm}
        OR LOWER(hook_script) LIKE ${searchTerm}
        OR LOWER(theme) LIKE ${searchTerm}
      ORDER BY day_of_year ASC
      LIMIT ${parseInt(limit)}
    `
    
    // Format results for display
    const lessons = results.map(row => ({
      day: row.day_of_year,
      topic: row.topic || 'Lesson',
      title: row.title,
      preview: row.summary || row.hook_script?.substring(0, 100),
      url: `/?day=${row.day_of_year}`
    }))
    
    return NextResponse.json({ 
      results: lessons,
      total: lessons.length,
      query
    })
  } catch (error) {
    console.error('Search error:', error)
    return NextResponse.json({ error: 'Search failed' }, { status: 500 })
  }
}
