// Route handler to serve Master Investment Thesis PDF
// Upload the actual PDF to Vercel Blob and update the URL below

import { redirect } from 'next/navigation'

export async function GET() {
  // TODO: Replace with actual Vercel Blob URL once PDF is uploaded
  // For now, redirect to strategic page with documents section
  redirect('/strategic#documents')
}
