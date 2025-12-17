/**
 * 🏭 FactoryDayView - v0 Template for Lotd
 * 
 * A 12×5 grid showing all archetypes × phases for a single lesson day.
 * Connects to Supabase to show real-time content/asset status.
 * 
 * Usage in v0:
 *   "Create a dashboard showing 12 Kelly archetypes across 5 lesson phases
 *    with status indicators for content, audio, and video assets"
 * 
 * Data sources:
 *   - core_lessons (topic, day_number)
 *   - lesson_atoms (content per archetype × phase)
 *   - kelly_video_assets (video generation status)
 *   - kelly-personas-manifest.json (archetype metadata)
 */

'use client';

import { useEffect, useState } from 'react';
import { createClient } from '@supabase/supabase-js';

// =============================================================================
// TYPES
// =============================================================================

interface Persona {
  id: string;
  name: string;
  icon: string;
  tagline: string;
  color: string;
}

interface AtomStatus {
  hasContent: boolean;
  hasAudio: boolean;
  hasVideo: boolean;
  videoStatus?: 'pending' | 'generating' | 'completed' | 'failed';
}

interface CellData {
  archetype: string;
  phase: string;
  status: AtomStatus;
  script?: string;
}

interface LessonData {
  dayNumber: number;
  topic: string;
  universalTruth: string;
  cells: CellData[];
}

// =============================================================================
// CONSTANTS - Your 12 Archetypes
// =============================================================================

const PERSONAS: Persona[] = [
  { id: 'scientist', name: 'The Scientist', icon: '🔬', tagline: 'Data-driven precision', color: '#3b82f6' },
  { id: 'explorer', name: 'The Explorer', icon: '🧭', tagline: 'Wonder and discovery', color: '#eab308' },
  { id: 'rebel', name: 'The Rebel', icon: '⚡', tagline: 'Bold challenging spirit', color: '#ef4444' },
  { id: 'architect', name: 'The Architect', icon: '🏛️', tagline: 'Methodical structure', color: '#6b7280' },
  { id: 'diplomat', name: 'The Diplomat', icon: '🤝', tagline: 'Inclusive harmony', color: '#22c55e' },
  { id: 'empath', name: 'The Empath', icon: '💗', tagline: 'Nurturing warmth', color: '#ec4899' },
  { id: 'macgyver', name: 'The MacGyver', icon: '🔧', tagline: 'Hands-on problem solver', color: '#f97316' },
  { id: 'mystic', name: 'The Mystic', icon: '✨', tagline: 'Profound serenity', color: '#a855f7' },
  { id: 'provider', name: 'The Provider', icon: '🛡️', tagline: 'Reassuring strength', color: '#14b8a6' },
  { id: 'storyteller', name: 'The Storyteller', icon: '📖', tagline: 'Theatrical captivation', color: '#f472b6' },
  { id: 'strategist', name: 'The Strategist', icon: '🎯', tagline: 'Sharp tactical mind', color: '#6366f1' },
  { id: 'survivor', name: 'The Survivor', icon: '🏕️', tagline: 'Grounded resilience', color: '#84cc16' },
];

const PHASES = ['Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom'] as const;

const PHASE_LABELS: Record<string, string> = {
  Hook: '🎣 Hook',
  Fact1: '1️⃣ Fact 1',
  Fact2: '2️⃣ Fact 2', 
  Fact3: '3️⃣ Fact 3',
  Wisdom: '🦉 Wisdom',
};

// =============================================================================
// SUPABASE CLIENT
// =============================================================================

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || 'https://tvjalxxsyryjphkforjv.supabase.co',
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || ''
);

// =============================================================================
// STATUS CELL COMPONENT
// =============================================================================

function StatusCell({ 
  status, 
  persona, 
  phase,
  onClick 
}: { 
  status: AtomStatus; 
  persona: Persona;
  phase: string;
  onClick?: () => void;
}) {
  const getStatusIcon = () => {
    if (status.hasVideo && status.videoStatus === 'completed') return '✅';
    if (status.videoStatus === 'generating') return '⏳';
    if (status.videoStatus === 'pending') return '🕐';
    if (status.videoStatus === 'failed') return '❌';
    if (status.hasAudio) return '🔊';
    if (status.hasContent) return '📝';
    return '⬜';
  };

  const getStatusColor = () => {
    if (status.hasVideo && status.videoStatus === 'completed') return 'bg-green-500/20 border-green-500/50';
    if (status.videoStatus === 'generating') return 'bg-yellow-500/20 border-yellow-500/50 animate-pulse';
    if (status.videoStatus === 'failed') return 'bg-red-500/20 border-red-500/50';
    if (status.hasAudio) return 'bg-blue-500/20 border-blue-500/50';
    if (status.hasContent) return 'bg-gray-500/20 border-gray-500/50';
    return 'bg-gray-900/50 border-gray-700/50';
  };

  return (
    <button
      onClick={onClick}
      className={`
        w-full h-12 rounded-lg border-2 
        ${getStatusColor()}
        flex items-center justify-center
        hover:scale-105 transition-all duration-200
        cursor-pointer group relative
      `}
      style={{ 
        boxShadow: status.hasVideo ? `0 0 12px ${persona.color}40` : 'none' 
      }}
    >
      <span className="text-lg">{getStatusIcon()}</span>
      
      {/* Hover tooltip */}
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 
                      opacity-0 group-hover:opacity-100 transition-opacity
                      bg-gray-900 text-white text-xs rounded px-2 py-1
                      whitespace-nowrap z-10 pointer-events-none">
        {persona.name} • {phase}
        <br />
        {status.hasContent ? '✓ Content' : '✗ Content'}
        {' | '}
        {status.hasAudio ? '✓ Audio' : '✗ Audio'}
        {' | '}
        {status.hasVideo ? '✓ Video' : '✗ Video'}
      </div>
    </button>
  );
}

// =============================================================================
// ARCHETYPE ROW COMPONENT
// =============================================================================

function ArchetypeRow({ 
  persona, 
  cells,
  onCellClick 
}: { 
  persona: Persona; 
  cells: CellData[];
  onCellClick?: (archetype: string, phase: string) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      {/* Archetype label */}
      <div 
        className="w-36 flex items-center gap-2 px-3 py-2 rounded-lg"
        style={{ backgroundColor: `${persona.color}20` }}
      >
        <span className="text-xl">{persona.icon}</span>
        <div className="flex flex-col">
          <span className="text-sm font-medium text-white truncate">
            {persona.name.replace('The ', '')}
          </span>
          <span className="text-xs text-gray-400 truncate">
            {persona.tagline}
          </span>
        </div>
      </div>
      
      {/* Phase cells */}
      <div className="flex-1 grid grid-cols-5 gap-2">
        {PHASES.map(phase => {
          const cell = cells.find(c => 
            c.archetype.toLowerCase().includes(persona.id) && 
            c.phase === phase
          );
          return (
            <StatusCell
              key={`${persona.id}-${phase}`}
              persona={persona}
              phase={phase}
              status={cell?.status || { hasContent: false, hasAudio: false, hasVideo: false }}
              onClick={() => onCellClick?.(persona.name, phase)}
            />
          );
        })}
      </div>
    </div>
  );
}

// =============================================================================
// STATS BAR COMPONENT
// =============================================================================

function StatsBar({ cells }: { cells: CellData[] }) {
  const total = 12 * 5; // 60 atoms per day
  const withContent = cells.filter(c => c.status.hasContent).length;
  const withAudio = cells.filter(c => c.status.hasAudio).length;
  const withVideo = cells.filter(c => c.status.hasVideo && c.status.videoStatus === 'completed').length;
  const generating = cells.filter(c => c.status.videoStatus === 'generating').length;
  const failed = cells.filter(c => c.status.videoStatus === 'failed').length;

  return (
    <div className="flex items-center gap-6 bg-gray-800/50 rounded-xl px-6 py-4">
      <Stat label="Total Atoms" value={total} icon="🎯" />
      <Stat label="Content" value={withContent} total={total} icon="📝" color="#6b7280" />
      <Stat label="Audio" value={withAudio} total={total} icon="🔊" color="#3b82f6" />
      <Stat label="Video" value={withVideo} total={total} icon="🎬" color="#22c55e" />
      {generating > 0 && <Stat label="Generating" value={generating} icon="⏳" color="#eab308" />}
      {failed > 0 && <Stat label="Failed" value={failed} icon="❌" color="#ef4444" />}
    </div>
  );
}

function Stat({ 
  label, 
  value, 
  total, 
  icon, 
  color 
}: { 
  label: string; 
  value: number; 
  total?: number; 
  icon: string;
  color?: string;
}) {
  const percentage = total ? Math.round((value / total) * 100) : null;
  
  return (
    <div className="flex items-center gap-3">
      <span className="text-2xl">{icon}</span>
      <div>
        <div className="text-xl font-bold text-white">
          {value}
          {total && <span className="text-gray-400 text-sm">/{total}</span>}
        </div>
        <div className="text-xs text-gray-400">{label}</div>
      </div>
      {percentage !== null && (
        <div 
          className="w-12 h-12 rounded-full flex items-center justify-center text-xs font-bold"
          style={{ 
            background: `conic-gradient(${color || '#22c55e'} ${percentage}%, transparent ${percentage}%)`,
          }}
        >
          <div className="w-9 h-9 rounded-full bg-gray-900 flex items-center justify-center">
            {percentage}%
          </div>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function FactoryDayView({ 
  dayNumber = 1 
}: { 
  dayNumber?: number 
}) {
  const [lesson, setLesson] = useState<LessonData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCell, setSelectedCell] = useState<{ archetype: string; phase: string } | null>(null);

  useEffect(() => {
    loadDayData(dayNumber);
  }, [dayNumber]);

  async function loadDayData(day: number) {
    setLoading(true);
    setError(null);

    try {
      // 1. Fetch core lesson
      const { data: coreLesson, error: lessonError } = await supabase
        .from('core_lessons')
        .select('id, day_number, topic, universal_truth')
        .eq('day_number', day)
        .single();

      if (lessonError) throw new Error(`Lesson not found: ${lessonError.message}`);

      // 2. Fetch all atoms for this lesson
      const { data: atoms, error: atomsError } = await supabase
        .from('lesson_atoms')
        .select('id, archetype, phase, content')
        .eq('core_lesson_id', coreLesson.id);

      if (atomsError) throw new Error(`Atoms error: ${atomsError.message}`);

      // 3. Fetch video assets for this day
      const { data: videos, error: videosError } = await supabase
        .from('kelly_video_assets')
        .select('archetype, phase, status, video_public_url')
        .eq('lesson_day', day);

      // Build cell data
      const cells: CellData[] = [];

      for (const persona of PERSONAS) {
        for (const phase of PHASES) {
          const atom = atoms?.find(a => 
            a.archetype?.toLowerCase().includes(persona.id) && 
            a.phase === phase
          );
          
          const video = videos?.find(v =>
            v.archetype?.toLowerCase().includes(persona.id) &&
            v.phase?.toLowerCase() === phase.toLowerCase()
          );

          cells.push({
            archetype: persona.name,
            phase,
            status: {
              hasContent: !!atom?.content,
              hasAudio: !!atom?.content?.script, // Assume audio if script exists
              hasVideo: !!video?.video_public_url,
              videoStatus: video?.status as AtomStatus['videoStatus'],
            },
            script: atom?.content?.script,
          });
        }
      }

      setLesson({
        dayNumber: coreLesson.day_number,
        topic: coreLesson.topic,
        universalTruth: coreLesson.universal_truth,
        cells,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }

  function handleCellClick(archetype: string, phase: string) {
    setSelectedCell({ archetype, phase });
    // Could open a modal, navigate, or trigger generation
    console.log(`Selected: ${archetype} × ${phase}`);
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl mb-4 animate-bounce">🏭</div>
          <div className="text-white text-lg">Loading Day {dayNumber}...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl mb-4">❌</div>
          <div className="text-red-400 text-lg">{error}</div>
          <button 
            onClick={() => loadDayData(dayNumber)}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!lesson) return null;

  return (
    <div className="min-h-screen bg-gray-950 text-white p-6">
      {/* Header */}
      <header className="mb-8">
        <div className="flex items-center gap-4 mb-2">
          <span className="text-3xl">🏭</span>
          <div>
            <h1 className="text-2xl font-bold">
              Day {lesson.dayNumber}: {lesson.topic}
            </h1>
            <p className="text-gray-400 text-sm max-w-2xl">
              {lesson.universalTruth}
            </p>
          </div>
        </div>
        
        {/* Day navigation */}
        <div className="flex items-center gap-2 mt-4">
          <button 
            onClick={() => loadDayData(Math.max(1, dayNumber - 1))}
            disabled={dayNumber <= 1}
            className="px-3 py-1 bg-gray-800 rounded hover:bg-gray-700 disabled:opacity-50"
          >
            ← Prev
          </button>
          <input
            type="number"
            min={1}
            max={365}
            value={dayNumber}
            onChange={(e) => loadDayData(Number(e.target.value))}
            className="w-20 px-3 py-1 bg-gray-800 rounded text-center"
          />
          <span className="text-gray-500">/ 365</span>
          <button 
            onClick={() => loadDayData(Math.min(365, dayNumber + 1))}
            disabled={dayNumber >= 365}
            className="px-3 py-1 bg-gray-800 rounded hover:bg-gray-700 disabled:opacity-50"
          >
            Next →
          </button>
        </div>
      </header>

      {/* Stats */}
      <StatsBar cells={lesson.cells} />

      {/* Phase Headers */}
      <div className="flex items-center gap-2 mt-8 mb-4">
        <div className="w-36" /> {/* Spacer for archetype column */}
        <div className="flex-1 grid grid-cols-5 gap-2">
          {PHASES.map(phase => (
            <div 
              key={phase}
              className="text-center text-sm font-medium text-gray-400 py-2"
            >
              {PHASE_LABELS[phase]}
            </div>
          ))}
        </div>
      </div>

      {/* Archetype Grid */}
      <div className="space-y-2">
        {PERSONAS.map(persona => (
          <ArchetypeRow
            key={persona.id}
            persona={persona}
            cells={lesson.cells}
            onCellClick={handleCellClick}
          />
        ))}
      </div>

      {/* Legend */}
      <div className="mt-8 flex items-center gap-6 text-sm text-gray-400">
        <span>Legend:</span>
        <span>⬜ Empty</span>
        <span>📝 Content</span>
        <span>🔊 Audio</span>
        <span>🕐 Pending</span>
        <span>⏳ Generating</span>
        <span>✅ Complete</span>
        <span>❌ Failed</span>
      </div>

      {/* Selected Cell Detail (placeholder for modal/panel) */}
      {selectedCell && (
        <div className="fixed bottom-6 right-6 bg-gray-800 rounded-xl p-4 shadow-xl border border-gray-700">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xl">
              {PERSONAS.find(p => p.name === selectedCell.archetype)?.icon}
            </span>
            <span className="font-medium">{selectedCell.archetype}</span>
            <span className="text-gray-400">×</span>
            <span>{selectedCell.phase}</span>
          </div>
          <div className="flex gap-2">
            <button className="px-3 py-1 bg-blue-600 rounded text-sm hover:bg-blue-700">
              View Content
            </button>
            <button className="px-3 py-1 bg-green-600 rounded text-sm hover:bg-green-700">
              Generate Video
            </button>
            <button 
              onClick={() => setSelectedCell(null)}
              className="px-3 py-1 bg-gray-700 rounded text-sm hover:bg-gray-600"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// USAGE EXAMPLES
// =============================================================================

/*
// In a Next.js page:
import FactoryDayView from '@/templates/v0/FactoryDayView';

export default function FactoryPage() {
  return <FactoryDayView dayNumber={1} />;
}

// With URL parameter:
import { useSearchParams } from 'next/navigation';

export default function FactoryPage() {
  const searchParams = useSearchParams();
  const day = Number(searchParams.get('day')) || 1;
  return <FactoryDayView dayNumber={day} />;
}
*/









