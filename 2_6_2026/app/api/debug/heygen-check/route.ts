import { NextRequest, NextResponse } from "next/server";
import { getSql } from "@/lib/db";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const day = parseInt(searchParams.get("day") || "34", 10);

  try {
    const sql = getSql()
    // Direct query to heygen_videos
    const result = await sql`
      SELECT id, day_of_year, phase, age_category, archetype, status, 
             video_url, created_at, updated_at
      FROM heygen_videos
      WHERE day_of_year = ${day}
        AND phase = 'hook'
        AND status IN ('completed', 'placeholder', 'ready')
        AND video_url IS NOT NULL
      ORDER BY updated_at DESC NULLS LAST
      LIMIT 5
    `;

    // Also get total count
    const countResult = await sql`
      SELECT COUNT(*) as total FROM heygen_videos 
      WHERE status = 'completed' AND video_url IS NOT NULL
    `;

    // Schema diagnostic - check which columns exist on critical tables
    const kellyColumns = await sql`
      SELECT column_name FROM information_schema.columns 
      WHERE table_name = 'kelly_lesson_assets' 
      ORDER BY ordinal_position
    `.catch(() => []);
    
    const perspectiveColumns = await sql`
      SELECT column_name FROM information_schema.columns 
      WHERE table_name = 'lesson_perspectives' 
      ORDER BY ordinal_position
    `.catch(() => []);
    
    // Check if the critical columns exist
    const kellyColNames = kellyColumns.map((c: Record<string, unknown>) => c.column_name as string);
    const perspColNames = perspectiveColumns.map((c: Record<string, unknown>) => c.column_name as string);
    
    const schemaCheck = {
      kelly_lesson_assets: {
        columns: kellyColNames,
        has_video_url: kellyColNames.includes('video_url'),
        has_audio_url: kellyColNames.includes('audio_url'),
        has_video_source: kellyColNames.includes('video_source'),
      },
      lesson_perspectives: {
        columns: perspColNames,
        has_subtitle: perspColNames.includes('subtitle'),
        has_hook_script: perspColNames.includes('hook_script'),
        has_title: perspColNames.includes('title'),
      },
    };

    return NextResponse.json({
      success: true,
      day,
      totalHeygenVideos: countResult[0]?.total || 0,
      rowsForDay: result.length,
      rows: result.map((r) => ({
        id: (r.id as string)?.substring(0, 8),
        age_category: r.age_category,
        archetype: r.archetype,
        status: r.status,
        video_url: (r.video_url as string)?.substring(0, 50),
        isHeygenUrl: (r.video_url as string)?.includes("heygen"),
        updated_at: r.updated_at,
      })),
      schemaCheck,
      databaseUrl: process.env.DATABASE_URL 
        ? process.env.DATABASE_URL.substring(0, 80) + "..." 
        : "NOT_SET",
      dbHost: process.env.DATABASE_URL?.match(/@([^/]+)/)?.[1] || "NOT_SET",
      neonBranch: process.env.DATABASE_URL?.match(/ep-([^.]+)/)?.[1] || "unknown",
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
        databaseUrl: process.env.DATABASE_URL 
          ? process.env.DATABASE_URL.substring(0, 80) + "..." 
          : "NOT_SET",
        dbHost: process.env.DATABASE_URL?.match(/@([^/]+)/)?.[1] || "NOT_SET",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
