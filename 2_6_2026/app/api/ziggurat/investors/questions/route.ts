import { NextRequest, NextResponse } from 'next/server'
import { sql } from '@/lib/db'

/**
 * Investor Questions API
 * 
 * POST /api/ziggurat/investors/questions - Add question to a model
 * POST /api/ziggurat/investors/questions/reply - Reply to a question
 */

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const modelId = searchParams.get('model_id')
    
    // If model_id provided, get questions for that model
    // Otherwise, get ALL questions with investor info (for admin dashboard)
    const questions = modelId
      ? await sql`
          SELECT * FROM investor_questions
          WHERE model_id = ${modelId}
          ORDER BY created_at DESC
        `
      : await sql`
          SELECT 
            q.*,
            m.investor_name,
            m.organization
          FROM investor_questions q
          LEFT JOIN investor_models m ON q.model_id = m.id
          ORDER BY q.created_at DESC
        `
    
    return NextResponse.json({ questions })
    
  } catch (error) {
    console.error('Error fetching questions:', error)
    return NextResponse.json({ error: 'Failed to fetch questions', questions: [] }, { status: 200 })
  }
}

// PATCH - Reply to a question
export async function PATCH(request: NextRequest) {
  try {
    const body = await request.json()
    const { questionId, reply } = body
    
    if (!questionId || !reply) {
      return NextResponse.json(
        { error: 'Missing required fields: questionId, reply' }, 
        { status: 400 }
      )
    }
    
    const result = await sql`
      UPDATE investor_questions 
      SET 
        reply = ${reply},
        replied_at = NOW()
      WHERE id = ${questionId}
      RETURNING *
    `
    
    if (result.length === 0) {
      return NextResponse.json({ error: 'Question not found' }, { status: 404 })
    }
    
    return NextResponse.json({ 
      success: true, 
      question: result[0] 
    })
    
  } catch (error) {
    console.error('Error replying to question:', error)
    return NextResponse.json({ error: 'Failed to reply to question' }, { status: 500 })
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { model_id, author, content } = body
    
    if (!model_id || !author || !content) {
      return NextResponse.json(
        { error: 'Missing required fields: model_id, author, content' }, 
        { status: 400 }
      )
    }
    
    const result = await sql`
      INSERT INTO investor_questions (
        model_id,
        author,
        content,
        created_at
      ) VALUES (
        ${model_id},
        ${author},
        ${content},
        NOW()
      )
      RETURNING *
    `
    
    // Update the model's updated_at timestamp
    await sql`
      UPDATE investor_models 
      SET updated_at = NOW()
      WHERE id = ${model_id}
    `
    
    return NextResponse.json({ 
      success: true, 
      question: result[0] 
    })
    
  } catch (error) {
    console.error('Error creating question:', error)
    return NextResponse.json({ error: 'Failed to create question' }, { status: 500 })
  }
}
