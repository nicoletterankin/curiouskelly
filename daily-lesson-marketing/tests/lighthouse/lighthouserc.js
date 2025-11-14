module.exports = {
  ci: {
    collect: {
      url: ['http://localhost:4321/'],
      numberOfRuns: 3,
      startServerCommand: 'npm run preview',
      startServerReadyPattern: 'Local',
      startServerReadyTimeout: 10000
    },
    assert: {
      assertions: {
        'categories:performance': ['error', { minScore: 0.9 }],
        'categories:accessibility': ['error', { minScore: 0.9 }],
        'categories:best-practices': ['error', { minScore: 0.9 }],
        'categories:seo': ['error', { minScore: 0.9 }],
        'first-contentful-paint': ['error', { maxNumericValue: 2000 }],
        'largest-contentful-paint': ['error', { maxNumericValue: 2500 }],
        'cumulative-layout-shift': ['error', { maxNumericValue: 0.1 }],
        'interactive': ['error', { maxNumericValue: 200 }],
        'total-blocking-time': ['error', { maxNumericValue: 200 }],
        'uses-responsive-images': 'error',
        'uses-optimized-images': 'error',
        'uses-webp-images': 'warn',
        'modern-image-formats': 'warn'
      }
    },
    upload: {
      target: 'temporary-public-storage'
    }
  }
};











