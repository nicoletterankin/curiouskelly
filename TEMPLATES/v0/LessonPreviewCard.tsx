/**
 * 📱 LessonPreviewCard - v0 Template for Lotd
 * 
 * A social-media-ready card showing a lesson preview.
 * Perfect for: landing pages, email, social media embeds.
 * 
 * Usage in v0:
 *   "Create a lesson preview card for social media that shows
 *    the day number, topic, Kelly avatar, and archetype badge"
 * 
 * Data sources:
 *   - core_lessons (topic, universal_truth)
 *   - kelly-personas-manifest.json (archetype visual)
 */

'use client';

import { useState, useEffect } from 'react';
import { 
  PERSONAS, 
  SUPABASE_CDN, 
  getPersonaImageUrl,
  type PersonaId,
  type CoreLesson,
} from './lib/personas';
import { fetchCoreLesson } from './lib/supabase';

// =============================================================================
// TYPES
// =============================================================================

interface LessonPreviewCardProps {
  /** Day number (1-365) - if provided, fetches from Supabase */
  dayNumber?: number;
  
  /** Or provide data directly */
  topic?: string;
  universalTruth?: string;
  
  /** Which archetype to display */
  archetypeId?: PersonaId;
  
  /** Visual variants */
  variant?: 'default' | 'compact' | 'hero' | 'social';
  
  /** Show "Today's Lesson" badge */
  showTodayBadge?: boolean;
  
  /** Click handler */
  onClick?: () => void;
  
  /** Additional classes */
  className?: string;
}

// =============================================================================
// COMPONENT
// =============================================================================

export default function LessonPreviewCard({
  dayNumber,
  topic: propTopic,
  universalTruth: propTruth,
  archetypeId = 'explorer',
  variant = 'default',
  showTodayBadge = false,
  onClick,
  className = '',
}: LessonPreviewCardProps) {
  const [lesson, setLesson] = useState<CoreLesson | null>(null);
  const [imageError, setImageError] = useState(false);

  // Fetch lesson data if dayNumber provided
  useEffect(() => {
    if (dayNumber && !propTopic) {
      fetchCoreLesson(dayNumber)
        .then(setLesson)
        .catch(console.error);
    }
  }, [dayNumber, propTopic]);

  const persona = PERSONAS[archetypeId];
  const topic = propTopic || lesson?.topic || 'Loading...';
  const universalTruth = propTruth || lesson?.universal_truth || '';
  const day = dayNumber || lesson?.day_number || 0;

  const imageUrl = getPersonaImageUrl(archetypeId, 'head');

  // ==========================================================================
  // VARIANT: COMPACT (small inline card)
  // ==========================================================================
  if (variant === 'compact') {
    return (
      <div
        onClick={onClick}
        className={`
          flex items-center gap-3 p-3 rounded-xl
          bg-gray-900/80 border border-gray-800
          ${onClick ? 'cursor-pointer hover:bg-gray-800/80' : ''}
          transition-all duration-200
          ${className}
        `}
      >
        {/* Kelly Avatar */}
        <div 
          className="w-10 h-10 rounded-full overflow-hidden flex-shrink-0 ring-2"
          style={{ ['--tw-ring-color' as string]: persona.color } as React.CSSProperties}
        >
          {!imageError ? (
            <img src={imageUrl} alt={persona.name} className="w-full h-full object-cover" onError={() => setImageError(true)} />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-lg" style={{ backgroundColor: `${persona.color}20` }}>
              {persona.icon}
            </div>
          )}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="text-xs text-gray-500">Day {day}</div>
          <div className="font-medium text-white truncate">{topic}</div>
        </div>

        {/* Arrow */}
        {onClick && (
          <div className="text-gray-500">→</div>
        )}
      </div>
    );
  }

  // ==========================================================================
  // VARIANT: SOCIAL (optimized for social media cards)
  // ==========================================================================
  if (variant === 'social') {
    return (
      <div
        className={`
          relative overflow-hidden rounded-2xl
          bg-gradient-to-br from-gray-900 to-gray-950
          border border-gray-800
          aspect-[1.91/1]
          ${className}
        `}
        style={{
          background: `linear-gradient(135deg, ${persona.color}10 0%, ${persona.color}05 50%, transparent 100%)`,
        }}
      >
        {/* Background glow */}
        <div 
          className="absolute top-0 right-0 w-1/2 h-full opacity-20"
          style={{
            background: `radial-gradient(circle at 80% 50%, ${persona.color}, transparent 60%)`,
          }}
        />

        {/* Content */}
        <div className="relative h-full flex items-center p-8">
          <div className="flex-1">
            {/* Brand */}
            <div className="flex items-center gap-2 mb-4">
              <span className="text-xl">✨</span>
              <span className="text-sm font-medium text-gray-400">Curious Kelly</span>
            </div>

            {/* Day */}
            <div 
              className="text-sm font-bold mb-2"
              style={{ color: persona.color }}
            >
              DAY {day}
            </div>

            {/* Topic */}
            <h2 className="text-2xl font-bold text-white mb-3">
              {topic}
            </h2>

            {/* Archetype badge */}
            <div 
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm"
              style={{ 
                backgroundColor: `${persona.color}20`,
                color: persona.color,
              }}
            >
              <span>{persona.icon}</span>
              <span>{persona.name}</span>
            </div>
          </div>

          {/* Kelly Avatar */}
          <div 
            className="w-32 h-32 rounded-full overflow-hidden ring-4 flex-shrink-0"
            style={{ 
              ['--tw-ring-color' as string]: persona.color,
              boxShadow: `0 0 40px ${persona.color}40`,
            }}
          >
            {!imageError ? (
              <img src={imageUrl} alt={persona.name} className="w-full h-full object-cover" onError={() => setImageError(true)} />
            ) : (
              <div 
                className="w-full h-full flex items-center justify-center text-4xl"
                style={{ backgroundColor: `${persona.color}20` }}
              >
                {persona.icon}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // ==========================================================================
  // VARIANT: HERO (large landing page card)
  // ==========================================================================
  if (variant === 'hero') {
    return (
      <div
        onClick={onClick}
        className={`
          relative overflow-hidden rounded-3xl
          bg-gradient-to-br from-gray-900 via-gray-900 to-gray-950
          border border-gray-800
          p-8 lg:p-12
          ${onClick ? 'cursor-pointer group' : ''}
          ${className}
        `}
      >
        {/* Background elements */}
        <div 
          className="absolute inset-0 opacity-10"
          style={{
            background: `radial-gradient(circle at 70% 30%, ${persona.color}, transparent 50%)`,
          }}
        />
        
        <div className="relative flex flex-col lg:flex-row items-center gap-8">
          {/* Kelly Avatar */}
          <div 
            className="w-40 h-40 lg:w-56 lg:h-56 rounded-full overflow-hidden ring-4 flex-shrink-0
                       group-hover:scale-105 transition-transform duration-300"
            style={{ 
              ['--tw-ring-color' as string]: persona.color,
              boxShadow: `0 0 60px ${persona.color}30`,
            }}
          >
            {!imageError ? (
              <img src={imageUrl} alt={persona.name} className="w-full h-full object-cover" onError={() => setImageError(true)} />
            ) : (
              <div 
                className="w-full h-full flex items-center justify-center text-6xl"
                style={{ backgroundColor: `${persona.color}20` }}
              >
                {persona.icon}
              </div>
            )}
          </div>

          {/* Content */}
          <div className="flex-1 text-center lg:text-left">
            {showTodayBadge && (
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-amber-500/20 to-orange-500/20 text-amber-400 text-sm font-medium mb-4">
                <span>🌟</span>
                <span>Today's Lesson</span>
              </div>
            )}

            <div className="text-gray-500 text-lg mb-2">Day {day} of 365</div>
            
            <h1 className="text-3xl lg:text-5xl font-bold text-white mb-4">
              {topic}
            </h1>

            <p className="text-gray-400 text-lg mb-6 max-w-2xl">
              {universalTruth}
            </p>

            {/* Archetype badge */}
            <div 
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-base"
              style={{ 
                backgroundColor: `${persona.color}20`,
                color: persona.color,
              }}
            >
              <span className="text-xl">{persona.icon}</span>
              <span className="font-medium">{persona.name}</span>
              <span className="text-gray-500">•</span>
              <span className="text-gray-400">{persona.tagline}</span>
            </div>
          </div>
        </div>

        {/* CTA */}
        {onClick && (
          <div className="mt-8 flex justify-center lg:justify-start">
            <button 
              className="px-8 py-4 rounded-xl font-bold text-lg transition-all duration-300
                         hover:scale-105 active:scale-95"
              style={{ 
                backgroundColor: persona.color,
                color: '#fff',
              }}
            >
              Start Learning →
            </button>
          </div>
        )}
      </div>
    );
  }

  // ==========================================================================
  // VARIANT: DEFAULT
  // ==========================================================================
  return (
    <div
      onClick={onClick}
      className={`
        relative overflow-hidden rounded-2xl
        bg-gray-900/80 backdrop-blur-sm
        border border-gray-800
        p-6
        ${onClick ? 'cursor-pointer hover:border-gray-700 hover:scale-[1.01]' : ''}
        transition-all duration-200
        ${className}
      `}
    >
      {/* Glow */}
      <div 
        className="absolute top-0 right-0 w-32 h-32 opacity-20 blur-2xl"
        style={{ backgroundColor: persona.color }}
      />

      <div className="relative flex items-start gap-5">
        {/* Kelly Avatar */}
        <div 
          className="w-20 h-20 rounded-full overflow-hidden ring-2 flex-shrink-0"
          style={{ 
            ['--tw-ring-color' as string]: persona.color,
            boxShadow: `0 0 20px ${persona.color}30`,
          }}
        >
          {!imageError ? (
            <img src={imageUrl} alt={persona.name} className="w-full h-full object-cover" onError={() => setImageError(true)} />
          ) : (
            <div 
              className="w-full h-full flex items-center justify-center text-2xl"
              style={{ backgroundColor: `${persona.color}20` }}
            >
              {persona.icon}
            </div>
          )}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {showTodayBadge && (
            <div className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 text-xs font-medium mb-2">
              <span>🌟</span>
              <span>Today</span>
            </div>
          )}

          <div className="text-sm text-gray-500 mb-1">Day {day}</div>
          
          <h3 className="text-xl font-bold text-white mb-2">
            {topic}
          </h3>

          <p className="text-gray-400 text-sm line-clamp-2 mb-3">
            {universalTruth}
          </p>

          {/* Archetype badge */}
          <div 
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-sm"
            style={{ 
              backgroundColor: `${persona.color}20`,
              color: persona.color,
            }}
          >
            <span>{persona.icon}</span>
            <span>{persona.name}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

















