module.exports = {
  ci: {
    collect: {
      startServerCommand: 'npm run build && npx serve dist',
      url: ['http://localhost:3000/', 'http://localhost:3000/adults/', 'http://localhost:3000/privacy/'],
      numberOfRuns: 1
    },
    assert: {
      preset: 'lighthouse:recommended',
      assertions: {
        'categories:performance': ['error', { minScore: 0.9 }],
        'largest-contentful-paint': ['error', { maxNumericValue: 2500 }],
        'cumulative-layout-shift': ['error', { maxNumericValue: 0.1 }],
        'interactive': ['error', { maxNumericValue: 4000 }],
        'total-byte-weight': [
          'error',
          {
            maxNumericValue: 410000
          }
        ],
        'unused-javascript': ['warn', { maxNumericValue: 150000 }],
        'uses-responsive-images': 'warn'
      }
    },
    upload: {
      target: 'filesystem',
      outputDir: 'tests/lighthouse/.lighthouseci'
    }
  }
};





