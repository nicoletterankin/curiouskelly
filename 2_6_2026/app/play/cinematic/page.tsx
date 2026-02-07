import { CinematicLesson } from "@/components/kelly/cinematic-lesson"
import { getDayNumber } from "@/lib/lessons"

export const metadata = {
  title: "Cinematic Lesson | The Daily Lesson",
  description: "Experience Kelly's lessons in full cinematic mode. Immersive audio, visual feedback, and seamless phase transitions.",
}

interface CinematicPlayPageProps {
  searchParams: Promise<{ 
    day?: string
    age?: string
    lang?: string
    archetype?: string
  }>
}

export default async function CinematicPlayPage({ searchParams }: CinematicPlayPageProps) {
  const params = await searchParams
  
  // Parse parameters
  const dayNumber = params.day ? parseInt(params.day, 10) : getDayNumber()
  const validDay = Math.min(365, Math.max(1, dayNumber || getDayNumber()))
  
  const ageGroup = params.age || 'adult'
  const language = params.lang || 'en'
  const archetype = params.archetype || 'explorer'
  
  return (
    <CinematicLesson
      dayNumber={validDay}
      ageGroup={ageGroup}
      language={language}
      archetype={archetype}
    />
  )
}
