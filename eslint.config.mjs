import js from "@eslint/js";
import prettierConfig from "eslint-config-prettier";
import tseslint from "@typescript-eslint/eslint-plugin";
import tsParser from "@typescript-eslint/parser";

const ignores = [
  "**/dist/**",
  "**/.turbo/**",
  "**/node_modules/**",
  "**/coverage/**",
  "**/.generated/**"
];

export default [
  {
    files: ["**/*.{ts,tsx,cts,mts}"],
    ignores,
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        project: "./tsconfig.base.json",
        tsconfigRootDir: process.cwd(),
        ecmaVersion: 2022,
        sourceType: "module"
      }
    },
    linterOptions: {
      reportUnusedDisableDirectives: true
    },
    plugins: {
      "@typescript-eslint": tseslint
    },
    rules: {
      ...tseslint.configs["recommended-type-checked"].rules,
      ...tseslint.configs["stylistic-type-checked"].rules,
      "@typescript-eslint/explicit-module-boundary-types": "off"
    }
  },
  {
    files: ["**/*.{js,cjs,mjs}"],
    ignores,
    languageOptions: {
      parserOptions: {
        ecmaVersion: 2022,
        sourceType: "module"
      }
    },
    rules: {
      ...js.configs.recommended.rules
    }
  },
  prettierConfig
];

