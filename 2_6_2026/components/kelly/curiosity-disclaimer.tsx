'use client'

import { Info } from 'lucide-react'

/**
 * Small disclaimer that appears on lesson pages
 * Reminds users that Kelly sparks curiosity but isn't always 100% accurate
 */
export function CuriosityDisclaimer() {
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground/70 px-3 py-1.5 bg-muted/30 rounded-full">
      <Info className="w-3 h-3 flex-shrink-0" />
      <span>Kelly sparks curiosity - keep exploring to learn more!</span>
    </div>
  )
}

/**
 * More detailed disclaimer for settings/about pages
 */
export function DetailedDisclaimer() {
  return (
    <div className="p-4 bg-muted/50 rounded-lg text-sm text-muted-foreground space-y-2">
      <div className="flex items-center gap-2 font-medium text-foreground">
        <Info className="w-4 h-4" />
        <span>About Kelly's Lessons</span>
      </div>
      <p>
        Kelly shares fascinating ideas to spark your curiosity and encourage learning. 
        While we strive for accuracy, science is always evolving and some topics have 
        multiple perspectives.
      </p>
      <p>
        We encourage you to explore further, ask questions, and discover more about 
        topics that interest you. That's what being curious is all about!
      </p>
    </div>
  )
}
