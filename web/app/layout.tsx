/**
 * Root layout for the PuntLab web dashboard.
 *
 * Purpose: define the document shell, metadata, and shared chrome for the
 * public-facing App Router experience.
 * Scope: wraps every route in the web application.
 * Dependencies: `app/globals.css`.
 */

import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";

import "./globals.css";

/**
 * Metadata shown in the browser tab and social previews.
 */
export const metadata: Metadata = {
  title: "PuntLab",
  description:
    "Daily accumulator recommendations powered by an autonomous research and scoring pipeline.",
};

/**
 * Shared layout shell for all web routes.
 *
 * @param props React props containing child route content.
 * @returns The root HTML structure for the PuntLab dashboard.
 */
export default function RootLayout({
  children,
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <div className="page-shell">
          <header className="site-header">
            <Link className="brand-mark" href="/">
              PuntLab
            </Link>
            <nav className="site-nav" aria-label="Primary navigation">
              <Link href="/">Home</Link>
              <Link href="/stats">Stats</Link>
              <Link href="/subscribe">Subscribe</Link>
              <Link className="nav-cta" href="/login">
                Dashboard
              </Link>
            </nav>
          </header>
          <main>{children}</main>
        </div>
      </body>
    </html>
  );
}
