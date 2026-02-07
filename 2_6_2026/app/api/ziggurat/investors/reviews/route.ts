import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * Investor Reviews API
 * 
 * GET  /api/ziggurat/investors/reviews - List scheduled reviews
 * POST /api/ziggurat/investors/reviews - Schedule a new review
 * PUT  /api/ziggurat/investors/reviews - Update review status
 */

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const modelId = searchParams.get('model_id')
    
    let reviews
    
    if (modelId) {
      reviews = await sql`
        SELECT * FROM investor_reviews
        WHERE model_id = ${modelId}
        ORDER BY scheduled_at ASC
      `
    } else {
      // Get all reviews (scheduled first, then by date)
      reviews = await sql`
        SELECT 
          r.id,
          r.model_id as investor_model_id,
          r.scheduled_at as scheduled_date,
          r.status,
          r.notes,
          m.investor_name,
          m.organization
        FROM investor_reviews r
        LEFT JOIN investor_models m ON r.model_id = m.id
        ORDER BY 
          CASE WHEN r.status = 'scheduled' THEN 0 ELSE 1 END,
          r.scheduled_at ASC
        LIMIT 50
      `.catch(() => [])
    }
    
    return NextResponse.json({ reviews })
    
  } catch (error) {
    console.error('Error fetching reviews:', error)
    return NextResponse.json({ error: 'Failed to fetch reviews' }, { status: 500 })
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { model_id, scheduled_at, attendees, notes } = body
    
    if (!model_id || !scheduled_at || !attendees) {
      return NextResponse.json(
        { error: 'Missing required fields: model_id, scheduled_at, attendees' }, 
        { status: 400 }
      )
    }
    
    const attendeesArray = Array.isArray(attendees) ? attendees : [attendees]
    
    const result = await sql`
      INSERT INTO investor_reviews (
        model_id,
        scheduled_at,
        attendees,
        notes,
        status,
        created_at
      ) VALUES (
        ${model_id},
        ${scheduled_at},
        ${attendeesArray},
        ${notes || ''},
        'scheduled',
        NOW()
      )
      RETURNING *
    `
    
    return NextResponse.json({ 
      success: true, 
      review: result[0] 
    })
    
  } catch (error) {
    console.error('Error scheduling review:', error)
    return NextResponse.json({ error: 'Failed to schedule review' }, { status: 500 })
  }
}

export async function PUT(request: NextRequest) {
  try {
    const body = await request.json()
    const { id, status, notes } = body
    
    if (!id || !status) {
      return NextResponse.json(
        { error: 'Missing required fields: id, status' }, 
        { status: 400 }
      )
    }
    
    const validStatuses = ['scheduled', 'completed', 'cancelled']
    if (!validStatuses.includes(status)) {
      return NextResponse.json(
        { error: 'Invalid status. Must be: scheduled, completed, or cancelled' }, 
        { status: 400 }
      )
    }
    
    const result = await sql`
      UPDATE investor_reviews
      SET 
        status = ${status},
        notes = COALESCE(${notes}, notes),
        updated_at = NOW()
      WHERE id = ${id}
      RETURNING *
    `
    
    if (result.length === 0) {
      return NextResponse.json({ error: 'Review not found' }, { status: 404 })
    }
    
    return NextResponse.json({ 
      success: true, 
      review: result[0] 
    })
    
  } catch (error) {
    console.error('Error updating review:', error)
    return NextResponse.json({ error: 'Failed to update review' }, { status: 500 })
  }
}
