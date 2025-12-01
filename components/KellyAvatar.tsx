'use client';

import { useState, useEffect } from 'react';
import Image from 'next/image';
import { getKellyAsset, KellyZoom, KellyPose } from '@/lib/kelly-assets';

interface KellyAvatarProps {
  zoom?: KellyZoom;
  pose?: KellyPose;
  className?: string;
  aspectRatio?: '21/9' | '16/9' | '4/3' | '9/16';
}

export function KellyAvatar({ 
  zoom = 'mid', 
  pose = 'left', 
  className = '',
  aspectRatio = '16/9'
}: KellyAvatarProps) {
  const [imgSrc, setImgSrc] = useState(getKellyAsset(zoom, pose));
  
  useEffect(() => {
    setImgSrc(getKellyAsset(zoom, pose));
  }, [zoom, pose]);

  return (
    <div 
      className={`kelly-avatar ${className}`}
      style={{ aspectRatio }}
    >
      <Image
        src={imgSrc}
        alt={`Kelly - ${pose}`}
        fill
        style={{ objectFit: 'cover', objectPosition: 'center' }}
        priority
      />
    </div>
  );
}
