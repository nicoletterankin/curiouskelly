module.exports = {
  root: true,
  env: {
    browser: true,
    es2022: true,
    node: true
  },
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    project: './tsconfig.json',
    extraFileExtensions: ['.astro']
  },
  plugins: [
    '@typescript-eslint',
    'astro',
    'import',
    'jsx-a11y',
    'unicorn'
  ],
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:astro/recommended',
    'plugin:import/recommended',
    'plugin:jsx-a11y/recommended',
    'plugin:unicorn/recommended',
    'prettier'
  ],
  settings: {
    'import/resolver': {
      node: true,
      typescript: {}
    }
  },
  rules: {
    'unicorn/prevent-abbreviations': 'off',
    'unicorn/filename-case': 'off',
    'import/no-unresolved': 'error',
    '@typescript-eslint/no-floating-promises': 'error'
  },
  overrides: [
    {
      files: ['*.astro'],
      parser: 'astro-eslint-parser',
      parserOptions: {
        parser: '@typescript-eslint/parser',
        extraFileExtensions: ['.astro']
      },
      rules: {
        'astro/jsx-a11y/anchor-is-valid': 'off'
      }
    },
    {
      files: ['tests/**/*.{ts,tsx}'],
      env: {
        node: true,
        browser: false
      }
    }
  ]
};




