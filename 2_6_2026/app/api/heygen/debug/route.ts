import { NextResponse } from 'next/server'

export async function GET() {
  const apiKey = process.env.HEYGEN_API_KEY
  
  // Check if key exists and show prefix (safe to log)
  const keyInfo = {
    exists: !!apiKey,
    length: apiKey?.length || 0,
    prefix: apiKey ? apiKey.substring(0, 15) + '...' : 'NOT SET',
    startsWithSk: apiKey?.startsWith('sk_') || false,
  }
  
  console.log('[HeyGen Debug] API Key info:', keyInfo)
  
  // Test the API directly
  if (apiKey) {
    try {
      const response = await fetch('https://api.heygen.com/v2/user/remaining_quota', {
        headers: {
          'X-Api-Key': apiKey,
          'Content-Type': 'application/json',
        },
      })
      
      const body = await response.text()
      console.log('[HeyGen Debug] API response:', response.status, body)
      
      return NextResponse.json({
        keyInfo,
        apiTest: {
          status: response.status,
          ok: response.ok,
          body: JSON.parse(body),
        }
      })
    } catch (error) {
      return NextResponse.json({
        keyInfo,
        apiTest: {
          error: error instanceof Error ? error.message : 'Unknown error'
        }
      })
    }
  }
  
  return NextResponse.json({ keyInfo, apiTest: null })
}
