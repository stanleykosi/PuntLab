/**
 * Next.js runtime configuration for the PuntLab web dashboard.
 *
 * Purpose: enable strict React rendering and lean production defaults for the
 * App Router shell.
 * Scope: framework-level behavior only; feature routing lives under `app/`.
 * Dependencies: consumed by `next dev`, `next build`, and `next start`.
 */

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
};

module.exports = nextConfig;
