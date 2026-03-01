import { redirect } from 'next/navigation'

// Redirect /zig to the slate vision page (LED concepts deprecated Feb 2026)
export default function ZigRedirect() {
  redirect('/ziggurat/slate')
}
