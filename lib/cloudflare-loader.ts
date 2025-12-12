/**
 * Cloudflare Image Resizing Loader for Next.js
 * 
 * Optimizes Kelly avatar images using Cloudflare's Image Resizing service.
 * Provides automatic format conversion, responsive sizing, and face-aware cropping.
 * 
 * Usage: Configure in next.config.js:
 * ```js
 * module.exports = {
 *   images: {
 *     loader: 'custom',
 *     loaderFile: './lib/cloudflare-loader.ts',
 *   },
 * }
 * ```
 */

import type { ImageLoaderProps } from "next/image";

export default function cloudflareLoader({ 
  src, 
  width, 
  quality = 80 
}: ImageLoaderProps): string {
  // Skip transformation in development
  if (process.env.NODE_ENV === "development") {
    return src;
  }
  
  // Build Cloudflare Image Resizing URL
  const params = [
    `width=${width}`,
    `quality=${quality}`,
    `format=auto`, // Auto-detect best format (WebP, AVIF)
    `fit=contain`, // Maintain aspect ratio
    `gravity=face` // Focus on Kelly's face when cropping
  ].join(",");
  
  // Cloudflare Image Resizing URL format
  // Assumes assets served from curiouskelly.com domain with R2 binding
  return `/cdn-cgi/image/${params}/${src.replace(/^\//, "")}`;
}

/**
 * Generate srcset for responsive images
 */
export function generateKellySrcSet(basePath: string): string {
  const widths = [320, 640, 768, 1024, 1280, 1536];
  return widths
    .map(w => `${cloudflareLoader({ src: basePath, width: w, quality: 80 })} ${w}w`)
    .join(", ");
}

/**
 * Get optimized Kelly image URL
 */
export function getKellyImageUrl(
  pose: string, 
  options: { 
    width?: number; 
    quality?: number; 
    variant?: string;
  } = {}
): string {
  const { width = 800, quality = 85, variant = "hero" } = options;
  const basePath = `/kelly/poses/${pose}/kelly_${pose}_${variant}.png`;
  return cloudflareLoader({ src: basePath, width, quality });
}
















