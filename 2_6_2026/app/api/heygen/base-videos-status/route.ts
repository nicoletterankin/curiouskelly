import { NextResponse } from "next/server";
import { getSql } from "@/lib/db";

/**
 * GET /api/heygen/base-videos-status
 *
 * Returns the current status of all 36 base avatar videos from the database.
 */
export async function GET() {
  try {
    const sql = getSql();
    // Get all base videos from database
    const videos = await sql`
      SELECT 
        age_category,
        archetype,
        avatar_key,
        heygen_video_id,
        status,
        video_url,
        thumbnail_url,
        duration_seconds,
        error_message,
        created_at,
        completed_at
      FROM heygen_videos
      WHERE video_type = 'base'
      ORDER BY age_category, archetype
    `;

    // Get stats
    const stats = await sql`SELECT * FROM get_heygen_stats()`;

    // Group by age category
    const byAge = {
      kid: videos.filter((v) => v.age_category === "kid"),
      adult: videos.filter((v) => v.age_category === "adult"),
      senior: videos.filter((v) => v.age_category === "senior"),
    };

    // Summary counts
    const summary = {
      total: videos.length,
      expected: 36,
      completed: videos.filter((v) => v.status === "completed").length,
      processing: videos.filter((v) => v.status === "processing").length,
      pending: videos.filter((v) => v.status === "pending").length,
      failed: videos.filter((v) => v.status === "failed").length,
      missing: 36 - videos.length,
    };

    // Get IDs of processing videos for easy polling
    const processingIds = videos
      .filter((v) => v.status === "processing")
      .map((v) => v.heygen_video_id)
      .filter(Boolean);

    return NextResponse.json({
      success: true,
      summary,
      stats: stats[0] || null,
      byAge: {
        kid: byAge.kid.map(formatVideo),
        adult: byAge.adult.map(formatVideo),
        senior: byAge.senior.map(formatVideo),
      },
      allVideos: videos.map(formatVideo),
      // Convenience for polling
      ...(processingIds.length > 0 && {
        pollUrl: `/api/heygen/check-base-videos?ids=${processingIds.join(",")}`,
      }),
      // If all complete, show the export
      ...(summary.completed === 36 && {
        exportReady: true,
        exportMessage:
          "All 36 base videos complete! Use GET /api/heygen/export-base-videos to get code snippet.",
      }),
    });
  } catch (error) {
    console.error("[HeyGen Base Status] Error:", error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to get status",
      },
      { status: 500 }
    );
  }
}

function formatVideo(v: Record<string, unknown>) {
  return {
    avatarKey: v.avatar_key,
    age: v.age_category,
    archetype: v.archetype,
    status: v.status,
    videoId: v.heygen_video_id,
    videoUrl: v.video_url,
    thumbnailUrl: v.thumbnail_url,
    duration: v.duration_seconds,
    error: v.error_message,
    createdAt: v.created_at,
    completedAt: v.completed_at,
  };
}
