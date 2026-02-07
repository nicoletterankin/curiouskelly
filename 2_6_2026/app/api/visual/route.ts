import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * GET /api/visual
 * Fetch visual aids for a specific day and phase
 * 
 * Query params:
 * - day: number (required) - day of year (1-365)
 * - phase: string (optional) - lesson phase (hook, story, wonder, action, wisdom)
 * - type: string (optional) - visual type filter
 */
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const day = parseInt(searchParams.get('day') || '0', 10)
  const phase = searchParams.get('phase')
  const type = searchParams.get('type')

  if (!day || day < 1 || day > 366) {
    return NextResponse.json(
      { error: 'Invalid day parameter. Must be 1-366.' },
      { status: 400 }
    )
  }

  try {
    // Try concept_assets first (primary source)
    let query = `
      SELECT 
        id,
        day,
        phase,
        visual_type,
        image_url,
        animation_url,
        prompt,
        approved,
        text_variants,
        visual_data,
        created_at
      FROM concept_assets
      WHERE day = $1
    `
    const params: (string | number)[] = [day]
    let paramIndex = 2

    if (phase) {
      query += ` AND phase = $${paramIndex}`
      params.push(phase)
      paramIndex++
    }

    if (type) {
      query += ` AND visual_type = $${paramIndex}`
      params.push(type)
      paramIndex++
    }

    query += ` ORDER BY phase, created_at DESC`

    const conceptAssets = await sql(query, params)

    // Also check lesson_perspectives for phase-specific visuals
    const perspectiveQuery = `
      SELECT 
        id,
        day_number as day,
        'hook' as phase, hook_visual as visual_url,
        'story' as phase2, story_visual as visual_url2,
        'wonder' as phase3, wonder_visual as visual_url3,
        'action' as phase4, action_visual as visual_url4,
        'wisdom' as phase5, wisdom_visual as visual_url5,
        title,
        age_group,
        archetype
      FROM lesson_perspectives
      WHERE day_number = $1
      LIMIT 1
    `
    const perspectives = await sql(perspectiveQuery, [day])

    // Also check generated_assets
    let generatedQuery = `
      SELECT 
        id,
        day_number as day,
        phase,
        asset_type as visual_type,
        url as image_url,
        prompt,
        status,
        created_at
      FROM generated_assets
      WHERE day_number = $1 AND asset_type = 'visual'
    `
    const generatedParams: (string | number)[] = [day]
    let genParamIndex = 2

    if (phase) {
      generatedQuery += ` AND phase = $${genParamIndex}`
      generatedParams.push(phase)
    }

    const generatedAssets = await sql(generatedQuery, generatedParams)

    // Combine and deduplicate results
    const allVisuals: Record<string, unknown>[] = []

    // Add concept assets
    for (const asset of conceptAssets) {
      allVisuals.push({
        id: asset.id,
        day: asset.day,
        phase: asset.phase,
        type: asset.visual_type,
        url: asset.image_url || asset.animation_url,
        prompt: asset.prompt,
        approved: asset.approved,
        source: 'concept_assets',
        variants: asset.text_variants,
        data: asset.visual_data,
      })
    }

    // Add generated assets if not already covered
    for (const asset of generatedAssets) {
      const exists = allVisuals.some(
        v => v.day === asset.day && v.phase === asset.phase
      )
      if (!exists && asset.image_url) {
        allVisuals.push({
          id: asset.id,
          day: asset.day,
          phase: asset.phase,
          type: asset.visual_type,
          url: asset.image_url,
          prompt: asset.prompt,
          approved: asset.status === 'complete',
          source: 'generated_assets',
        })
      }
    }

    // Parse perspectives and add if no visuals found
    if (perspectives.length > 0) {
      const p = perspectives[0]
      const phaseVisualMap: Record<string, string | null> = {
        hook: p.hook_visual as string | null,
        story: p.story_visual as string | null,
        wonder: p.wonder_visual as string | null,
        action: p.action_visual as string | null,
        wisdom: p.wisdom_visual as string | null,
      }

      for (const [phaseName, url] of Object.entries(phaseVisualMap)) {
        if (url && (!phase || phase === phaseName)) {
          const exists = allVisuals.some(
            v => v.day === day && v.phase === phaseName
          )
          if (!exists) {
            allVisuals.push({
              id: `perspective-${day}-${phaseName}`,
              day,
              phase: phaseName,
              type: 'perspective',
              url,
              approved: true,
              source: 'lesson_perspectives',
              title: p.title,
              ageGroup: p.age_group,
              archetype: p.archetype,
            })
          }
        }
      }
    }

    // Sort by phase order
    const phaseOrder = ['hook', 'story', 'wonder', 'action', 'wisdom']
    allVisuals.sort((a, b) => {
      const aIndex = phaseOrder.indexOf(a.phase as string)
      const bIndex = phaseOrder.indexOf(b.phase as string)
      return aIndex - bIndex
    })

    return NextResponse.json({
      day,
      phase: phase || 'all',
      visuals: allVisuals,
      count: allVisuals.length,
      hasVisuals: allVisuals.length > 0,
    })

  } catch (error) {
    console.error('[api/visual] Error fetching visuals:', error)
    return NextResponse.json(
      { error: 'Failed to fetch visual aids', details: String(error) },
      { status: 500 }
    )
  }
}
