import { redirect } from 'next/navigation'

// Redirect /zig to /ziggurat - single source of truth for the LED vision project
export default function ZigRedirect() {
  redirect('/ziggurat')
}
