/**
 * ADMIN: Scan all lessons to find days where scripts don't match the corrected topic
 * 
 * GET /api/admin/regenerate-scripts/scan
 * Returns a list of days that need script regeneration
 */

import { NextResponse } from "next/server"
import { getSql } from "@/lib/db"
import curriculumLookup from "@/data/curriculum_lookup.json"

const CURRICULUM: Record<
  string,
  { topic: string; date: string; learning_objective: string }
> = curriculumLookup

// Known contamination patterns - scripts that clearly reference wrong topics
const CONTAMINATION_MARKERS = [
  "dolphin",
  "deja vu",
  "why we dream",
  "how dolphins",
  "animal sleep",
  "sleep with only half",
  "lucid dream",
  "why do we yawn",
  "why is the sky",
  "hiccup",
]

export async function GET() {
  try {
    const sql = getSql()

    const lessons = await sql`
      SELECT day_of_year, title, topic, hook_script, story_script, wonder_script, action_script, wisdom_script, hook_fact, content_version
      FROM lessons 
      ORDER BY day_of_year
    `

    const misaligned: Array<{
      day: number
      title: string
      topic: string
      curriculum_topic: string
      issue: string
      hook_preview: string
    }> = []

    const aligned: number[] = []
    const missing: number[] = []

    for (const lesson of lessons as Array<Record<string, unknown>>) {
      const day = lesson.day_of_year as number
      const curriculum = CURRICULUM[String(day)]

      if (!curriculum) {
        missing.push(day)
        continue
      }

      const hookScript = ((lesson.hook_script as string) || "").toLowerCase()
      const storyScript = ((lesson.story_script as string) || "").toLowerCase()
      const wonderScript = ((lesson.wonder_script as string) || "").toLowerCase()
      const allScripts = `${hookScript} ${storyScript} ${wonderScript}`

      // Check for contamination markers
      const contaminated = CONTAMINATION_MARKERS.some((marker) =>
        allScripts.includes(marker),
      )

      // Check if scripts reference the correct topic
      const topicWords = (curriculum.topic || "")
        .toLowerCase()
        .split(/[\s\-–—]+/)
        .filter((w) => w.length > 4)
        .filter(
          (w) =>
            ![
              "about",
              "their",
              "these",
              "those",
              "which",
              "where",
              "there",
              "being",
            ].includes(w),
        )

      const topicMatch = topicWords.some(
        (word) =>
          hookScript.includes(word) ||
          storyScript.includes(word) ||
          wonderScript.includes(word),
      )

      // No scripts at all
      const noScripts = !lesson.hook_script && !lesson.story_script

      if (contaminated) {
        misaligned.push({
          day,
          title: lesson.title as string,
          topic: lesson.topic as string,
          curriculum_topic: curriculum.topic,
          issue: "contaminated",
          hook_preview: (lesson.hook_script as string || "").slice(0, 80),
        })
      } else if (noScripts) {
        misaligned.push({
          day,
          title: lesson.title as string,
          topic: lesson.topic as string,
          curriculum_topic: curriculum.topic,
          issue: "no_scripts",
          hook_preview: "",
        })
      } else if (!topicMatch) {
        misaligned.push({
          day,
          title: lesson.title as string,
          topic: lesson.topic as string,
          curriculum_topic: curriculum.topic,
          issue: "likely_misaligned",
          hook_preview: (lesson.hook_script as string || "").slice(0, 80),
        })
      } else {
        aligned.push(day)
      }
    }

    return NextResponse.json({
      summary: {
        total_lessons: lessons.length,
        aligned: aligned.length,
        misaligned: misaligned.length,
        missing_from_curriculum: missing.length,
        contaminated: misaligned.filter((m) => m.issue === "contaminated")
          .length,
        no_scripts: misaligned.filter((m) => m.issue === "no_scripts").length,
        likely_misaligned: misaligned.filter(
          (m) => m.issue === "likely_misaligned",
        ).length,
      },
      misaligned_days: misaligned,
      aligned_days: aligned,
    })
  } catch (error) {
    console.error("[admin/regenerate-scripts/scan] Error:", error)
    return NextResponse.json({ error: String(error) }, { status: 500 })
  }
}
