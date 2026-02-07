import { NextResponse } from 'next/server'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
}

export async function OPTIONS() {
  return NextResponse.json({}, { headers: corsHeaders })
}

export async function GET() {
  return NextResponse.json(
    { 
      ok: true, 
      ts: new Date().toISOString(),
      env: process.env.VERCEL_ENV || 'development',
    },
    { headers: corsHeaders }
  )
}
