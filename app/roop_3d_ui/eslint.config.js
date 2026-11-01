// Syntax-level lint for .js/.jsx/.ts/.tsx. typescript-eslint cannot be used: TypeScript 7 ships no compiler JS API
// (and typescript-eslint requires <6.1), so types are checked by `tsc --noEmit` and this config parses with Babel.
import js from '@eslint/js';
import globals from 'globals';
import babelParser from '@babel/eslint-parser';
import reactHooks from 'eslint-plugin-react-hooks';

export default [
  { ignores: ['dist/**', 'node_modules/**'] },
  js.configs.recommended,
  {
    files: ['**/*.{js,jsx,ts,tsx}'],
    languageOptions: {
      parser: babelParser,
      parserOptions: {
        requireConfigFile: false,
        babelOptions: { presets: [['@babel/preset-react', { runtime: 'automatic' }], '@babel/preset-typescript'] },
      },
      globals: { ...globals.browser, ...globals.node },
    },
    plugins: { 'react-hooks': reactHooks },
    rules: {
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',
      'no-undef': 'off',            // TypeScript reports undefined names (and DOM/JSX type names) precisely
      'no-unused-vars': ['warn', { argsIgnorePattern: '^_', varsIgnorePattern: '^_', caughtErrors: 'none' }],
      'no-empty': ['error', { allowEmptyCatch: true }],
    },
  },
  {
    // Babel's scope analysis cannot see TypeScript type-only usage, so unused-variable reports are false positives in .ts/.tsx.
    files: ['**/*.{ts,tsx}'],
    rules: { 'no-unused-vars': 'off' },
  },
];
