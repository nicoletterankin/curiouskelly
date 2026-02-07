import { NextResponse } from 'next/server'

// Redirect to the markdown file hosted on Vercel Blob
export async function GET() {
  return NextResponse.redirect(
    'https://blobs.vusercontent.net/blob/V0_CURRICULUM_AUTHORITY_DIRECTIVE-qhXb40qxU5Fut25teRR9jStGeiX9qd.md',
    { status: 302 }
  )
}
