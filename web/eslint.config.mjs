/**
 * ESLint configuration for the PuntLab web dashboard.
 *
 * Purpose: apply Next.js App Router, TypeScript, and Core Web Vitals rules to
 * the web surface without bringing in Tailwind-specific presets.
 * Scope: lint rules only; formatting stays editor-driven.
 * Dependencies: ESLint, `eslint-config-next`, and `@eslint/eslintrc`.
 */

import { FlatCompat } from "@eslint/eslintrc";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    ignores: ["node_modules/**", ".next/**", "out/**", "build/**", "next-env.d.ts"],
  },
];

export default eslintConfig;
