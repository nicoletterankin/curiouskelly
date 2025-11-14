module.exports = {
  extends: ['stylelint-config-standard-scss', 'stylelint-config-prettier-scss'],
  rules: {
    'selector-class-pattern': null,
    'color-function-notation': 'legacy',
    'alpha-value-notation': 'number'
  },
  ignoreFiles: ['public/**/*', 'dist/**/*']
};





