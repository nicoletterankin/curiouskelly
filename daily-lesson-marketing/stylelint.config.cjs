module.exports = {
  extends: ['stylelint-config-standard-scss'],
  rules: {
    'selector-class-pattern': null,
    'scss/at-import-partial-extension': null,
    'property-no-vendor-prefix': null,
    'value-no-vendor-prefix': null
  },
  ignoreFiles: ['node_modules/**', 'dist/**', '.astro/**']
};











