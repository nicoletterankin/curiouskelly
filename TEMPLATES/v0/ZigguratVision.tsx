/**
 * 🏛️ ZigguratVision - LED Visualization Component
 * 
 * Displays the Chet Holifield Federal Building with LED lighting.
 * Can be used standalone or embedded in the right rail.
 * 
 * Assets live in /public/ziggurat/precision/
 * 
 * Usage:
 *   <ZigguratVision palette="gold" time="night" />
 *   <ZigguratVision variant="rainbow-twilight" compact />
 */

'use client';

import { useState, useCallback } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type ZigguratPalette = 'rainbow' | 'cool' | 'warm' | 'white' | 'gold' | 'cyan' | 'usa';
export type ZigguratTime = 'night' | 'late-night' | 'twilight' | 'dusk';
export type ZigguratResolution = 'thumb' | '1080p' | '4k' | 'full';

interface ZigguratVisionProps {
  /** Palette name */
  palette?: ZigguratPalette;
  /** Time of day */
  time?: ZigguratTime;
  /** Resolution to display */
  resolution?: ZigguratResolution;
  /** Combined variant string (overrides palette+time) */
  variant?: string;
  /** Compact mode for embedding */
  compact?: boolean;
  /** Show before/after slider */
  showSlider?: boolean;
  /** CSS class for container */
  className?: string;
  /** Callback when variant changes */
  onVariantChange?: (palette: ZigguratPalette, time: ZigguratTime) => void;
}

// =============================================================================
// CONSTANTS
// =============================================================================

export const PALETTES: Record<ZigguratPalette, { label: string; colors: string[] }> = {
  rainbow: { label: 'Rainbow', colors: ['#9333ea', '#3b82f6', '#06b6d4', '#22c55e', '#eab308', '#f97316', '#ef4444'] },
  cool: { label: 'Cool', colors: ['#9333ea', '#6366f1', '#3b82f6', '#0ea5e9', '#06b6d4', '#14b8a6', '#10b981'] },
  warm: { label: 'Warm', colors: ['#fbbf24', '#f59e0b', '#f97316', '#ef4444', '#ec4899', '#d946ef', '#a855f7'] },
  white: { label: 'White', colors: ['#fffdf8', '#fffdf8', '#fffdf8', '#fffdf8', '#fffdf8', '#fffdf8', '#fffdf8'] },
  gold: { label: 'Gold', colors: ['#ffd700', '#ffc700', '#ffb700', '#ffa500', '#ff9500', '#ff8500', '#ff7500'] },
  cyan: { label: 'Cyan', colors: ['#00ffff', '#00f0fa', '#00e0f0', '#00d0e0', '#00c0d0', '#00b0c0', '#00a0b0'] },
  usa: { label: 'USA', colors: ['#ef4444', '#ffffff', '#3b82f6', '#ffffff', '#ef4444', '#ffffff', '#3b82f6'] },
};

export const TIMES: Record<ZigguratTime, { label: string; icon: string }> = {
  night: { label: 'Night', icon: '🌙' },
  'late-night': { label: 'Late Night', icon: '🌑' },
  twilight: { label: 'Twilight', icon: '🌆' },
  dusk: { label: 'Dusk', icon: '🌅' },
};

const CDN_BASE = '/ziggurat/precision';

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

export function getZigguratUrl(
  palette: ZigguratPalette,
  time: ZigguratTime,
  resolution: ZigguratResolution = '1080p'
): string {
  return `${CDN_BASE}/${palette}-${time}-${resolution}.jpg`;
}

export function getBeforeUrl(resolution: ZigguratResolution = '1080p'): string {
  return `${CDN_BASE}/before-${resolution}.jpg`;
}

// =============================================================================
// COMPACT BUTTON (for right rail)
// =============================================================================

export function ZigguratButton({
  onClick,
  className = '',
}: {
  onClick?: () => void;
  className?: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`
        relative w-full aspect-video rounded-lg overflow-hidden
        border border-white/10 hover:border-purple-500/50
        transition-all duration-200 group
        ${className}
      `}
      aria-label="View Ziggurat LED Vision"
    >
      <img
        src={getZigguratUrl('gold', 'night', 'thumb')}
        alt="Ziggurat LED Vision"
        className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity"
      />
      <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />
      <span className="absolute bottom-1 left-1 text-[10px] text-white/80 font-medium">
        🏛️ Ziggurat
      </span>
    </button>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function ZigguratVision({
  palette: initialPalette = 'gold',
  time: initialTime = 'night',
  resolution = '1080p',
  variant,
  compact = false,
  showSlider = true,
  className = '',
  onVariantChange,
}: ZigguratVisionProps) {
  // Parse variant string if provided
  const parsedPalette = variant?.split('-')[0] as ZigguratPalette || initialPalette;
  const parsedTime = variant?.split('-').slice(1).join('-') as ZigguratTime || initialTime;

  const [palette, setPalette] = useState<ZigguratPalette>(parsedPalette);
  const [time, setTime] = useState<ZigguratTime>(parsedTime);
  const [sliderPos, setSliderPos] = useState(50);
  const [isDragging, setIsDragging] = useState(false);

  const handlePaletteChange = useCallback((p: ZigguratPalette) => {
    setPalette(p);
    onVariantChange?.(p, time);
  }, [time, onVariantChange]);

  const handleTimeChange = useCallback((t: ZigguratTime) => {
    setTime(t);
    onVariantChange?.(palette, t);
  }, [palette, onVariantChange]);

  const handleSliderMove = useCallback((clientX: number, rect: DOMRect) => {
    const pos = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
    setSliderPos(pos);
  }, []);

  // Compact mode: just the image
  if (compact) {
    return (
      <div className={`relative rounded-lg overflow-hidden ${className}`}>
        <img
          src={getZigguratUrl(palette, time, 'thumb')}
          alt={`Ziggurat - ${PALETTES[palette].label} ${TIMES[time].label}`}
          className="w-full aspect-video object-cover"
        />
      </div>
    );
  }

  return (
    <div className={`bg-gray-950 rounded-xl overflow-hidden ${className}`}>
      {/* Image with slider */}
      <div
        className="relative cursor-ew-resize select-none"
        onMouseDown={(e) => {
          setIsDragging(true);
          handleSliderMove(e.clientX, e.currentTarget.getBoundingClientRect());
        }}
        onMouseMove={(e) => {
          if (isDragging) {
            handleSliderMove(e.clientX, e.currentTarget.getBoundingClientRect());
          }
        }}
        onMouseUp={() => setIsDragging(false)}
        onMouseLeave={() => setIsDragging(false)}
        onTouchStart={(e) => {
          setIsDragging(true);
          handleSliderMove(e.touches[0].clientX, e.currentTarget.getBoundingClientRect());
        }}
        onTouchMove={(e) => {
          if (isDragging) {
            handleSliderMove(e.touches[0].clientX, e.currentTarget.getBoundingClientRect());
          }
        }}
        onTouchEnd={() => setIsDragging(false)}
      >
        {/* After image (full) */}
        <img
          src={getZigguratUrl(palette, time, resolution)}
          alt={`Ziggurat - ${PALETTES[palette].label} ${TIMES[time].label}`}
          className="w-full aspect-video object-cover"
        />

        {/* Before image (clipped) */}
        {showSlider && (
          <>
            <img
              src={getBeforeUrl(resolution)}
              alt="Ziggurat - Before"
              className="absolute inset-0 w-full h-full object-cover"
              style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
            />
            
            {/* Slider line */}
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg pointer-events-none"
              style={{ left: `${sliderPos}%`, transform: 'translateX(-50%)' }}
            >
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 
                            bg-white text-black px-2 py-1 rounded-full text-xs font-bold shadow-xl">
                ⟷
              </div>
            </div>

            {/* Labels */}
            <span className="absolute top-3 left-3 bg-black/70 backdrop-blur px-2 py-1 rounded text-[10px] text-white font-medium">
              BEFORE
            </span>
            <span className="absolute top-3 right-3 bg-black/70 backdrop-blur px-2 py-1 rounded text-[10px] text-white font-medium">
              {PALETTES[palette].label.toUpperCase()} · {TIMES[time].label.toUpperCase()}
            </span>
          </>
        )}
      </div>

      {/* Controls */}
      <div className="p-3 bg-gray-900/80 border-t border-white/5">
        <div className="flex gap-2">
          {/* Palette selector */}
          <select
            value={palette}
            onChange={(e) => handlePaletteChange(e.target.value as ZigguratPalette)}
            className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs text-white"
            aria-label="Select palette"
          >
            {Object.entries(PALETTES).map(([key, { label }]) => (
              <option key={key} value={key}>{label}</option>
            ))}
          </select>

          {/* Time selector */}
          <select
            value={time}
            onChange={(e) => handleTimeChange(e.target.value as ZigguratTime)}
            className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs text-white"
            aria-label="Select time of day"
          >
            {Object.entries(TIMES).map(([key, { label, icon }]) => (
              <option key={key} value={key}>{icon} {label}</option>
            ))}
          </select>
        </div>

        {/* Palette preview */}
        <div className="flex gap-0.5 mt-2">
          {PALETTES[palette].colors.map((color, i) => (
            <div
              key={i}
              className="flex-1 h-1 rounded-full"
              style={{ backgroundColor: color }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// EXPORTS
// =============================================================================

export { ZigguratVision };
