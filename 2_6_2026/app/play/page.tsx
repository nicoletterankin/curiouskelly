import KellyPlayer from "@/components/kelly/kelly-player"
import { getDayNumber } from "@/lib/lessons"

export const metadata = {
  title: "Curious Kelly | Lesson of the Day",
  description: "AI-powered education for every learner. A new lesson every day, personalized to your age, language, and learning style.",
}

interface PlayPageProps {
  searchParams: Promise<{ day?: string; phase?: string }>
}

export default async function PlayPage({ searchParams }: PlayPageProps) {
  const params = await searchParams
  
  // Support ?day=27 or default to today's lesson
  const dayParam = params.day
  const dayNumber = dayParam ? parseInt(dayParam, 10) : getDayNumber()
  
  // Validate day is 1-365
  const validDay = Math.min(365, Math.max(1, dayNumber || getDayNumber()))
  
  // Support ?phase=hook|story|wonder|action|wisdom
  const initialPhase = params.phase as "hook" | "story" | "wonder" | "action" | "wisdom" | undefined
  
  return <KellyPlayer dayNumber={validDay} initialPhase={initialPhase} />
}
