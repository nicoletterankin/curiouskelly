import type { MetadataRoute } from 'next'

/**
 * Dynamic sitemap generation for The Daily Lesson
 * Includes all 365 lesson URLs for search engine crawling
 */

export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = 'https://thedailylesson.com'
  
  // Core pages
  const corePages: MetadataRoute.Sitemap = [
    {
      url: baseUrl,
      lastModified: new Date(),
      changeFrequency: 'daily',
      priority: 1.0,
    },
    {
      url: `${baseUrl}/curriculum`,
      lastModified: new Date(),
      changeFrequency: 'monthly',
      priority: 0.9,
    },
    {
      url: `${baseUrl}/strategic`,
      lastModified: new Date(),
      changeFrequency: 'weekly',
      priority: 0.8,
    },
    {
      url: `${baseUrl}/about`,
      lastModified: new Date(),
      changeFrequency: 'monthly',
      priority: 0.7,
    },
    {
      url: `${baseUrl}/docs`,
      lastModified: new Date(),
      changeFrequency: 'weekly',
      priority: 0.7,
    },
    {
      url: `${baseUrl}/coalition`,
      lastModified: new Date(),
      changeFrequency: 'weekly',
      priority: 0.8,
    },
    {
      url: `${baseUrl}/products/ilearn`,
      lastModified: new Date(),
      changeFrequency: 'monthly',
      priority: 0.7,
    },
  ]
  
  // Generate all 365 lesson URLs
  const lessonPages: MetadataRoute.Sitemap = Array.from({ length: 365 }, (_, i) => ({
    url: `${baseUrl}/?day=${i + 1}`,
    lastModified: new Date(),
    changeFrequency: 'yearly' as const,
    priority: 0.6,
  }))
  
  return [...corePages, ...lessonPages]
}
