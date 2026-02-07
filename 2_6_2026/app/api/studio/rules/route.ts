import { NextResponse } from 'next/server'
import { sql } from '@/lib/db'

// GET - Fetch all active Rankin Rules
export async function GET() {
  try {
    const rules = await sql`
      SELECT id, rule_text, source_session, source_date, applies_to, priority, is_active
      FROM rankin_rules 
      WHERE is_active = true
      ORDER BY priority DESC, created_at ASC
    `
    
    return NextResponse.json({ rules })
  } catch (error) {
    console.error('[studio/rules] GET error:', error)
    return NextResponse.json({ error: 'Failed to fetch rules' }, { status: 500 })
  }
}

// POST - Create a new rule from Nicolette's decision
export async function POST(request: Request) {
  try {
    const { rule_text, source_session, applies_to = ['all'] } = await request.json()
    
    if (!rule_text) {
      return NextResponse.json({ error: 'rule_text is required' }, { status: 400 })
    }
    
    const result = await sql`
      INSERT INTO rankin_rules (rule_text, source_session, applies_to, created_by)
      VALUES (${rule_text}, ${source_session || 'Studio session'}, ${applies_to}, 'nicolette')
      RETURNING id, rule_text, source_session, applies_to, created_at
    `
    
    console.log('[studio/rules] New Rankin Rule created:', rule_text)
    
    return NextResponse.json({ 
      success: true, 
      rule: result[0],
      message: `Kelly will now remember: "${rule_text}"`
    })
  } catch (error) {
    console.error('[studio/rules] POST error:', error)
    return NextResponse.json({ error: 'Failed to create rule' }, { status: 500 })
  }
}

// DELETE - Deactivate a rule (soft delete)
export async function DELETE(request: Request) {
  try {
    const { id } = await request.json()
    
    if (!id) {
      return NextResponse.json({ error: 'id is required' }, { status: 400 })
    }
    
    await sql`
      UPDATE rankin_rules 
      SET is_active = false 
      WHERE id = ${id}::uuid
    `
    
    return NextResponse.json({ success: true })
  } catch (error) {
    console.error('[studio/rules] DELETE error:', error)
    return NextResponse.json({ error: 'Failed to delete rule' }, { status: 500 })
  }
}
