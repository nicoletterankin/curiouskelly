// HTTP Execution Tool - Allows v0 to execute API calls via neon_run_sql logging
// This creates a record in the database that can be polled for results

import { neon, NeonQueryFunction } from '@neondatabase/serverless'
import { NextRequest, NextResponse } from 'next/server'

// Lazy-initialized SQL client - prevents build-time connection errors
let _sql: NeonQueryFunction<false, false> | null = null
function getSql() {
  if (!_sql) {
    _sql = neon(process.env.DATABASE_URL!)
  }
  return _sql
}

// Create execution log table if not exists
async function ensureTable() {
  const sql = getSql()
  await sql`
    CREATE TABLE IF NOT EXISTS api_executions (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      endpoint TEXT NOT NULL,
      method TEXT DEFAULT 'POST',
      payload JSONB,
      status TEXT DEFAULT 'pending',
      response JSONB,
      error TEXT,
      created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
      completed_at TIMESTAMP WITH TIME ZONE
    )
  `
}

export async function POST(request: NextRequest) {
  const { endpoint, method = 'POST', payload, execute = false } = await request.json()
  
  if (!endpoint) {
    return NextResponse.json({ error: 'endpoint required' }, { status: 400 })
  }
  
  await ensureTable()
  
  // If execute is false, just log what WOULD be executed
  if (!execute) {
    return NextResponse.json({
      dryRun: true,
      wouldExecute: {
        endpoint,
        method,
        payload
      },
      message: 'Set execute: true to actually run this'
    })
  }
  
  // Log the execution request
  const sql = getSql()
  const [logEntry] = await sql`
    INSERT INTO api_executions (endpoint, method, payload, status)
    VALUES (${endpoint}, ${method}, ${JSON.stringify(payload)}::jsonb, 'running')
    RETURNING id
  `
  
  try {
    // Build the full URL
    const baseUrl = process.env.VERCEL_URL 
      ? `https://${process.env.VERCEL_URL}` 
      : 'http://localhost:3000'
    const fullUrl = endpoint.startsWith('http') ? endpoint : `${baseUrl}${endpoint}`
    
    // Execute the request
    const response = await fetch(fullUrl, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: method !== 'GET' ? JSON.stringify(payload) : undefined
    })
    
    const responseText = await response.text()
    let responseData
    try {
      responseData = JSON.parse(responseText)
    } catch {
      responseData = { text: responseText }
    }
    
    // Update log with result
    await sql`
      UPDATE api_executions 
      SET status = 'completed', 
          response = ${JSON.stringify(responseData)}::jsonb,
          completed_at = NOW()
      WHERE id = ${logEntry.id}
    `
    
    return NextResponse.json({
      success: true,
      executionId: logEntry.id,
      response: responseData
    })
    
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error'
    
    await sql`
      UPDATE api_executions 
      SET status = 'failed', 
          error = ${errorMsg},
          completed_at = NOW()
      WHERE id = ${logEntry.id}
    `
    
    return NextResponse.json({
      success: false,
      executionId: logEntry.id,
      error: errorMsg
    }, { status: 500 })
  }
}

// GET recent executions
export async function GET() {
  await ensureTable()
  
  const sql = getSql()
  const executions = await sql`
    SELECT * FROM api_executions 
    ORDER BY created_at DESC 
    LIMIT 20
  `
  
  return NextResponse.json({ executions })
}
