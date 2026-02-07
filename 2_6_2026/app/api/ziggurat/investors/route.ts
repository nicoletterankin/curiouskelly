import { NextRequest, NextResponse } from 'next/server'
import { neon, NeonQueryFunction } from '@neondatabase/serverless'

// Lazy-initialized SQL client - prevents build-time connection errors
let _sql: NeonQueryFunction<false, false> | null = null
function getSql() {
  if (!_sql) {
    _sql = neon(process.env.DATABASE_URL!)
  }
  return _sql
}

/**
 * Investor Models API
 * 
 * Endpoints:
 * GET  /api/ziggurat/investors         - List all public investor models
 * POST /api/ziggurat/investors         - Create a new investor model
 * GET  /api/ziggurat/investors/[id]    - Get specific investor model
 * PUT  /api/ziggurat/investors/[id]    - Update investor model inputs
 * POST /api/ziggurat/investors/[id]/questions - Add question to model
 * POST /api/ziggurat/investors/[id]/reviews   - Schedule a review
 */

export async function GET(request: NextRequest) {
  const sql = getSql()
  try {
    const { searchParams } = new URL(request.url)
    const investorId = searchParams.get('id')
    
    if (investorId) {
      // Get specific investor model
      const models = await sql`
        SELECT * FROM investor_models 
        WHERE id = ${investorId}
      `
      
      if (models.length === 0) {
        return NextResponse.json({ error: 'Model not found' }, { status: 404 })
      }
      
      // Get questions for this model
      const questions = await sql`
        SELECT * FROM investor_questions 
        WHERE model_id = ${investorId}
        ORDER BY created_at DESC
      `
      
      // Get scheduled reviews
      const reviews = await sql`
        SELECT * FROM investor_reviews
        WHERE model_id = ${investorId}
        ORDER BY scheduled_at ASC
      `
      
      // Increment view count
      await sql`
        UPDATE investor_models 
        SET view_count = view_count + 1 
        WHERE id = ${investorId}
      `
      
      return NextResponse.json({
        model: models[0],
        questions,
        reviews
      })
    }
    
    // List all public models
    const models = await sql`
      SELECT 
        id,
        investor_name,
        organization,
        created_at,
        updated_at,
        is_public,
        view_count,
        notes
      FROM investor_models 
      WHERE is_public = true
      ORDER BY updated_at DESC
    `
    
    return NextResponse.json({ models })
    
  } catch (error) {
    console.error('Error fetching investor models:', error)
    return NextResponse.json({ error: 'Failed to fetch models' }, { status: 500 })
  }
}

export async function POST(request: NextRequest) {
  const sql = getSql()
  try {
    const body = await request.json()
    const { 
      investor_name, 
      investor_email, 
      organization, 
      notes,
      is_public = true,
      model_inputs = {}
    } = body
    
    if (!investor_name || !investor_email || !organization) {
      return NextResponse.json(
        { error: 'Missing required fields' }, 
        { status: 400 }
      )
    }
    
    const result = await sql`
      INSERT INTO investor_models (
        investor_name,
        investor_email,
        organization,
        notes,
        is_public,
        model_inputs,
        view_count,
        created_at,
        updated_at
      ) VALUES (
        ${investor_name},
        ${investor_email},
        ${organization},
        ${notes || ''},
        ${is_public},
        ${JSON.stringify(model_inputs)},
        0,
        NOW(),
        NOW()
      )
      RETURNING *
    `
    
    return NextResponse.json({ 
      success: true, 
      model: result[0] 
    })
    
  } catch (error) {
    console.error('Error creating investor model:', error)
    return NextResponse.json({ error: 'Failed to create model' }, { status: 500 })
  }
}

export async function PUT(request: NextRequest) {
  const sql = getSql()
  try {
    const body = await request.json()
    const { id, model_inputs, notes } = body
    
    if (!id) {
      return NextResponse.json({ error: 'Model ID required' }, { status: 400 })
    }
    
    const result = await sql`
      UPDATE investor_models 
      SET 
        model_inputs = ${JSON.stringify(model_inputs)},
        notes = COALESCE(${notes}, notes),
        updated_at = NOW()
      WHERE id = ${id}
      RETURNING *
    `
    
    if (result.length === 0) {
      return NextResponse.json({ error: 'Model not found' }, { status: 404 })
    }
    
    return NextResponse.json({ 
      success: true, 
      model: result[0] 
    })
    
  } catch (error) {
    console.error('Error updating investor model:', error)
    return NextResponse.json({ error: 'Failed to update model' }, { status: 500 })
  }
}
