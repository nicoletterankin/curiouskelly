import { NextResponse } from "next/server";
import { getSql } from "@/lib/db";

/**
 * GET /api/heygen/export-base-videos
 *
 * Exports completed base videos as TypeScript code for kelly-assets.ts
 */
export async function GET() {
  try {
    const sql = getSql();
    const videos = await sql`
      SELECT 
        age_category,
        archetype,
        avatar_key,
        video_url,
        thumbnail_url,
        duration_seconds
      FROM heygen_videos
      WHERE video_type = 'base' AND status = 'completed'
      ORDER BY age_category, archetype
    `;

    if (videos.length === 0) {
      return NextResponse.json({
        success: false,
        error: "No completed base videos found",
        hint: "Run POST /api/heygen/kickoff-base-videos first",
      });
    }

    // Generate TypeScript export
    const tsCode = generateTypeScriptExport(videos);

    // Generate JSON for direct use
    const jsonExport: Record<
      string,
      { url: string; thumbnail: string; duration: number }
    > = {};
    for (const v of videos) {
      jsonExport[v.avatar_key as string] = {
        url: v.video_url as string,
        thumbnail: (v.thumbnail_url as string) || "",
        duration: Number(v.duration_seconds) || 0,
      };
    }

    return NextResponse.json({
      success: true,
      count: videos.length,
      complete: videos.length === 36,
      missing: 36 - videos.length,
      // TypeScript code to paste into kelly-assets.ts
      typescript: tsCode,
      // JSON format for direct use
      json: jsonExport,
      // Instructions
      instructions: [
        "1. Copy the 'typescript' field content",
        "2. Paste into /lib/kelly-assets.ts",
        "3. Export the HEYGEN_BASE_VIDEOS constant",
        "4. Use getBaseVideoUrl(age, archetype) to retrieve URLs",
      ],
    });
  } catch (error) {
    console.error("[HeyGen Export] Error:", error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Export failed",
      },
      { status: 500 }
    );
  }
}

function generateTypeScriptExport(
  videos: Array<Record<string, unknown>>
): string {
  const lines: string[] = [
    "/**",
    " * HeyGen Base Videos - Auto-generated",
    ` * Generated: ${new Date().toISOString()}`,
    ` * Total: ${videos.length} videos`,
    " */",
    "export const HEYGEN_BASE_VIDEOS: Record<string, {",
    "  url: string;",
    "  thumbnail: string;",
    "  duration: number;",
    "}> = {",
  ];

  for (const v of videos) {
    const key = v.avatar_key as string;
    const url = v.video_url as string;
    const thumb = (v.thumbnail_url as string) || "";
    const duration = Number(v.duration_seconds) || 0;

    lines.push(`  "${key}": {`);
    lines.push(`    url: "${url}",`);
    lines.push(`    thumbnail: "${thumb}",`);
    lines.push(`    duration: ${duration},`);
    lines.push("  },");
  }

  lines.push("};");
  lines.push("");
  lines.push("/**");
  lines.push(" * Get base video URL for a specific avatar");
  lines.push(" */");
  lines.push(
    "export function getBaseVideoUrl(age: 'kid' | 'adult' | 'senior', archetype: string): string | null {"
  );
  lines.push('  const key = `${age}_${archetype}`;');
  lines.push("  return HEYGEN_BASE_VIDEOS[key]?.url || null;");
  lines.push("}");

  return lines.join("\n");
}
