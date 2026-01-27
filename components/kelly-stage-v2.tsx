'use client';

import { useState, useEffect } from 'react';
import Image from 'next/image';

// Kelly static image - using LoRA-generated curious pose
const KELLY_STATIC_IMAGE = "/kelly/curious.png";

// Alternative: Use Supabase CDN if available
// const KELLY_STATIC_IMAGE = "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/heygen/archetypes-head-only/kelly_scientist_head.png";

interface KellyStageV2Props {
  className?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  showFallback?: boolean;
}

export default function KellyStageV2({
  className = '',
  size = 'md',
  showFallback = true,
}: KellyStageV2Props) {
  const [imageError, setImageError] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);

  const sizeClasses = {
    sm: 'w-32 h-32',
    md: 'w-64 h-64',
    lg: 'w-96 h-96',
    xl: 'w-[512px] h-[512px]',
  };

  return (
    <div className={`relative ${sizeClasses[size]} ${className}`}>
      {!imageError ? (
        <img
          src={KELLY_STATIC_IMAGE}
          alt="Kelly - AI Learning Companion"
          className="w-full h-full object-cover rounded-lg"
          onLoad={() => setImageLoaded(true)}
          onError={() => {
            console.error(`[KellyStageV2] Failed to load image: ${KELLY_STATIC_IMAGE}`);
            setImageError(true);
          }}
          style={{
            opacity: imageLoaded ? 1 : 0,
            transition: 'opacity 0.3s ease-in-out',
          }}
        />
      ) : showFallback ? (
        <div className="w-full h-full flex items-center justify-center bg-gray-800 rounded-lg">
          <div className="text-center">
            <div className="text-4xl mb-2">✨</div>
            <div className="text-sm text-gray-400">Kelly</div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
