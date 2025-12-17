/**
 * 🎭 ArchetypeCard - v0 Template for Lotd
 * 
 * A card component that adapts to any of the 12 Kelly archetypes.
 * Shows the archetype's Kelly image, colors, and styling.
 * 
 * Usage in v0:
 *   "Create a lesson card that shows Kelly's avatar, the archetype name,
 *    lesson topic, and adapts its color scheme to the archetype"
 * 
 * Data sources:
 *   - kelly-personas-manifest.json (archetype metadata)
 *   - Supabase CDN for Kelly images
 */

'use client';

import { useState } from 'react';

// =============================================================================
// TYPES
// =============================================================================

interface Persona {
  id: string;
  name: string;
  icon: string;
  tagline: string;
  description: string;
  color: string;
  images: {
    head: string;
    clean: string;
    prop: string;
  };
}

type PersonaId = 
  | 'scientist' | 'explorer' | 'rebel' | 'architect'
  | 'diplomat' | 'empath' | 'macgyver' | 'mystic'
  | 'provider' | 'storyteller' | 'strategist' | 'survivor';

interface ArchetypeCardProps {
  archetypeId: PersonaId;
  topic?: string;
  dayNumber?: number;
  subtitle?: string;
  showImage?: boolean;
  imageVariant?: 'head' | 'clean' | 'prop';
  size?: 'sm' | 'md' | 'lg';
  onClick?: () => void;
  selected?: boolean;
  className?: string;
}

// =============================================================================
// CONSTANTS
// =============================================================================

const SUPABASE_CDN = 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates';

const PERSONAS: Record<PersonaId, Persona> = {
  scientist: {
    id: 'scientist',
    name: 'The Scientist',
    icon: '🔬',
    tagline: 'Data-driven precision',
    description: 'Lab goggles on forehead',
    color: '#3b82f6',
    images: {
      head: 'heygen/archetypes-head-only/kelly_scientist_head.png',
      clean: 'heygen/archetypes-head-only/kelly_scientist_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_scientist_prop_head.png',
    },
  },
  explorer: {
    id: 'explorer',
    name: 'The Explorer',
    icon: '🧭',
    tagline: 'Wonder and discovery',
    description: 'Aviator goggles + bandana',
    color: '#eab308',
    images: {
      head: 'heygen/archetypes-head-only/kelly_explorer_head.png',
      clean: 'heygen/archetypes-head-only/kelly_explorer_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_explorer_prop_head.png',
    },
  },
  rebel: {
    id: 'rebel',
    name: 'The Rebel',
    icon: '⚡',
    tagline: 'Bold challenging spirit',
    description: 'Sunglasses in hair + earring',
    color: '#ef4444',
    images: {
      head: 'heygen/archetypes-head-only/kelly_rebel_head.png',
      clean: 'heygen/archetypes-head-only/kelly_rebel_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_rebel_prop_head.png',
    },
  },
  architect: {
    id: 'architect',
    name: 'The Architect',
    icon: '🏛️',
    tagline: 'Methodical structure',
    description: 'Pencil behind ear + glasses',
    color: '#6b7280',
    images: {
      head: 'heygen/archetypes-head-only/kelly_architect_head.png',
      clean: 'heygen/archetypes-head-only/kelly_architect_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_architect_prop_head.png',
    },
  },
  diplomat: {
    id: 'diplomat',
    name: 'The Diplomat',
    icon: '🤝',
    tagline: 'Inclusive harmony',
    description: 'Pearl studs + velvet headband',
    color: '#22c55e',
    images: {
      head: 'heygen/archetypes-head-only/kelly_diplomat_head.png',
      clean: 'heygen/archetypes-head-only/kelly_diplomat_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_diplomat_prop_head.png',
    },
  },
  empath: {
    id: 'empath',
    name: 'The Empath',
    icon: '💗',
    tagline: 'Nurturing warmth',
    description: 'Pink headband + lavender',
    color: '#ec4899',
    images: {
      head: 'heygen/archetypes-head-only/kelly_empath_head.png',
      clean: 'heygen/archetypes-head-only/kelly_empath_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_empath_prop_head.png',
    },
  },
  macgyver: {
    id: 'macgyver',
    name: 'The MacGyver',
    icon: '🔧',
    tagline: 'Hands-on problem solver',
    description: 'Shop glasses + red bandana',
    color: '#f97316',
    images: {
      head: 'heygen/archetypes-head-only/kelly_macgyver_head.png',
      clean: 'heygen/archetypes-head-only/kelly_macgyver_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_macgyver_prop_head.png',
    },
  },
  mystic: {
    id: 'mystic',
    name: 'The Mystic',
    icon: '✨',
    tagline: 'Profound serenity',
    description: 'Third eye amethyst + gold chain',
    color: '#a855f7',
    images: {
      head: 'heygen/archetypes-head-only/kelly_mystic_head.png',
      clean: 'heygen/archetypes-head-only/kelly_mystic_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_mystic_prop_head.png',
    },
  },
  provider: {
    id: 'provider',
    name: 'The Provider',
    icon: '🛡️',
    tagline: 'Reassuring strength',
    description: 'Cream knit headband',
    color: '#14b8a6',
    images: {
      head: 'heygen/archetypes-head-only/kelly_provider_head.png',
      clean: 'heygen/archetypes-head-only/kelly_provider_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_provider_prop_head.png',
    },
  },
  storyteller: {
    id: 'storyteller',
    name: 'The Storyteller',
    icon: '📖',
    tagline: 'Theatrical captivation',
    description: 'Gold glasses + peacock feather',
    color: '#f472b6',
    images: {
      head: 'heygen/archetypes-head-only/kelly_storyteller_head.png',
      clean: 'heygen/archetypes-head-only/kelly_storyteller_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_storyteller_prop_head.png',
    },
  },
  strategist: {
    id: 'strategist',
    name: 'The Strategist',
    icon: '🎯',
    tagline: 'Sharp tactical mind',
    description: 'Angular glasses + chess clip',
    color: '#6366f1',
    images: {
      head: 'heygen/archetypes-head-only/kelly_strategist_head.png',
      clean: 'heygen/archetypes-head-only/kelly_strategist_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_strategist_prop_head.png',
    },
  },
  survivor: {
    id: 'survivor',
    name: 'The Survivor',
    icon: '🏕️',
    tagline: 'Grounded resilience',
    description: 'Military bandana + dog tags',
    color: '#84cc16',
    images: {
      head: 'heygen/archetypes-head-only/kelly_survivor_head.png',
      clean: 'heygen/archetypes-head-only/kelly_survivor_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_survivor_prop_head.png',
    },
  },
};

// =============================================================================
// COMPONENT
// =============================================================================

export default function ArchetypeCard({
  archetypeId,
  topic,
  dayNumber,
  subtitle,
  showImage = true,
  imageVariant = 'head',
  size = 'md',
  onClick,
  selected = false,
  className = '',
}: ArchetypeCardProps) {
  const [imageError, setImageError] = useState(false);
  const persona = PERSONAS[archetypeId];

  if (!persona) {
    console.error(`Unknown archetype: ${archetypeId}`);
    return null;
  }

  const sizeClasses = {
    sm: 'p-3 rounded-lg',
    md: 'p-4 rounded-xl',
    lg: 'p-6 rounded-2xl',
  };

  const imageSizes = {
    sm: 'w-12 h-12',
    md: 'w-16 h-16',
    lg: 'w-24 h-24',
  };

  const textSizes = {
    sm: { name: 'text-sm', tagline: 'text-xs', topic: 'text-base' },
    md: { name: 'text-base', tagline: 'text-sm', topic: 'text-lg' },
    lg: { name: 'text-lg', tagline: 'text-base', topic: 'text-xl' },
  };

  const imageUrl = `${SUPABASE_CDN}/${persona.images[imageVariant]}`;

  return (
    <div
      onClick={onClick}
      className={`
        ${sizeClasses[size]}
        ${onClick ? 'cursor-pointer hover:scale-[1.02] active:scale-[0.98]' : ''}
        ${selected ? 'ring-2 ring-offset-2 ring-offset-gray-950' : ''}
        transition-all duration-200
        bg-gray-900/80 backdrop-blur-sm
        border border-gray-800
        ${className}
      `}
      style={{
        borderColor: selected ? persona.color : undefined,
        boxShadow: selected ? `0 0 20px ${persona.color}30` : undefined,
        '--ring-color': persona.color,
      } as React.CSSProperties}
    >
      <div className="flex items-start gap-4">
        {/* Kelly Avatar */}
        {showImage && (
          <div 
            className={`
              ${imageSizes[size]}
              rounded-full overflow-hidden
              bg-gradient-to-br from-gray-800 to-gray-900
              flex-shrink-0
              ring-2
            `}
            style={{ 
              ['--tw-ring-color' as string]: persona.color,
              boxShadow: `0 0 20px ${persona.color}40`,
            } as React.CSSProperties}
          >
            {!imageError ? (
              <img
                src={imageUrl}
                alt={persona.name}
                className="w-full h-full object-cover"
                onError={() => setImageError(true)}
              />
            ) : (
              <div 
                className="w-full h-full flex items-center justify-center text-2xl"
                style={{ backgroundColor: `${persona.color}20` }}
              >
                {persona.icon}
              </div>
            )}
          </div>
        )}

        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Archetype badge */}
          <div className="flex items-center gap-2 mb-1">
            <span className="text-lg">{persona.icon}</span>
            <span 
              className={`${textSizes[size].name} font-medium`}
              style={{ color: persona.color }}
            >
              {persona.name}
            </span>
          </div>

          {/* Tagline */}
          <p className={`${textSizes[size].tagline} text-gray-400 mb-2`}>
            {persona.tagline}
          </p>

          {/* Topic (if provided) */}
          {topic && (
            <h3 className={`${textSizes[size].topic} font-bold text-white truncate`}>
              {dayNumber && <span className="text-gray-500">Day {dayNumber}: </span>}
              {topic}
            </h3>
          )}

          {/* Subtitle (if provided) */}
          {subtitle && (
            <p className={`${textSizes[size].tagline} text-gray-400 mt-1 line-clamp-2`}>
              {subtitle}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// GRID VARIANT - Shows all 12 archetypes
// =============================================================================

export function ArchetypeGrid({
  selectedId,
  onSelect,
  size = 'sm',
}: {
  selectedId?: PersonaId;
  onSelect?: (id: PersonaId) => void;
  size?: 'sm' | 'md';
}) {
  const archetypeIds = Object.keys(PERSONAS) as PersonaId[];

  return (
    <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
      {archetypeIds.map(id => {
        const persona = PERSONAS[id];
        return (
          <button
            key={id}
            onClick={() => onSelect?.(id)}
            className={`
              p-3 rounded-xl text-center
              transition-all duration-200
              hover:scale-105 active:scale-95
              ${selectedId === id 
                ? 'ring-2 ring-offset-2 ring-offset-gray-950' 
                : 'opacity-70 hover:opacity-100'
              }
            `}
            style={{
              backgroundColor: `${persona.color}20`,
              borderColor: selectedId === id ? persona.color : 'transparent',
              boxShadow: selectedId === id ? `0 0 20px ${persona.color}40` : undefined,
            }}
          >
            <div className="text-3xl mb-2">{persona.icon}</div>
            <div 
              className="text-sm font-medium truncate"
              style={{ color: persona.color }}
            >
              {persona.name.replace('The ', '')}
            </div>
          </button>
        );
      })}
    </div>
  );
}

// =============================================================================
// BADGE VARIANT - Compact pill
// =============================================================================

export function ArchetypeBadge({
  archetypeId,
  showIcon = true,
}: {
  archetypeId: PersonaId;
  showIcon?: boolean;
}) {
  const persona = PERSONAS[archetypeId];
  if (!persona) return null;

  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-sm font-medium"
      style={{ 
        backgroundColor: `${persona.color}20`,
        color: persona.color,
      }}
    >
      {showIcon && <span>{persona.icon}</span>}
      <span>{persona.name}</span>
    </span>
  );
}

// =============================================================================
// EXPORTS
// =============================================================================

export { PERSONAS, SUPABASE_CDN };
export type { PersonaId, Persona, ArchetypeCardProps };









