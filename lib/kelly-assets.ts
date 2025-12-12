export type KellyZoom = 'out' | 'mid' | 'in';
export type KellyPose = 'left' | 'right' | 'up' | 'down' | 'neutral' | 'thinking' | 'correct' | 'incorrect' | 'hint';

export const KELLY_ASSETS: Record<KellyZoom, Partial<Record<KellyPose, string>>> = {
  out: {
    left: '/kelly/out/out-left.jpeg',
    right: '/kelly/out/out-right.jpeg',
  },
  mid: {
    left: '/kelly/mid/mid-left.jpeg',
    right: '/kelly/mid/mid-right.jpeg',
  },
  in: {
    left: '/kelly/in/in-left.jpeg',
    right: '/kelly/in/in-right.jpeg',
  },
};

// Fallback logic: if pose missing, try neutral, then left
export function getKellyAsset(zoom: KellyZoom, pose: KellyPose): string {
  const zoomAssets = KELLY_ASSETS[zoom];
  if (zoomAssets[pose]) return zoomAssets[pose]!;
  if (zoomAssets.neutral) return zoomAssets.neutral;
  if (zoomAssets.left) return zoomAssets.left!;
  return KELLY_ASSETS.mid.left!; // ultimate fallback
}

// For portrait mode, map left/right to up/down (we'll CSS flip for now)
export function getKellyPoseForOrientation(
  pose: KellyPose, 
  orientation: 'landscape' | 'portrait'
): { actualPose: KellyPose; flip: boolean } {
  if (orientation === 'portrait') {
    if (pose === 'left') return { actualPose: 'left', flip: false }; // TODO: use 'up' when available
    if (pose === 'right') return { actualPose: 'right', flip: false }; // TODO: use 'down' when available
  }
  return { actualPose: pose, flip: false };
}














