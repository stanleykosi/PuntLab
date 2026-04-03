/**
 * Supabase client helpers for the PuntLab web dashboard.
 *
 * Purpose: expose browser and server factories with strict environment
 * validation so later auth and data-fetching code share one canonical setup.
 * Scope: public anon-key clients only; privileged server clients belong in the
 * API or backend services.
 * Dependencies: `@supabase/ssr`, `@supabase/supabase-js`, and `next/headers`.
 */

import { createBrowserClient, createServerClient } from "@supabase/ssr";
import type { CookieOptions } from "@supabase/ssr";
import { cookies } from "next/headers";

/**
 * Minimal database typing placeholder until the generated Supabase types are
 * introduced in a later implementation step.
 */
export interface Database {
  public: {
    Tables: Record<string, never>;
    Views: Record<string, never>;
    Functions: Record<string, never>;
    Enums: Record<string, never>;
    CompositeTypes: Record<string, never>;
  };
}

interface SupabaseEnvironment {
  readonly url: string;
  readonly anonKey: string;
}

interface SerializableCookie {
  readonly name: string;
  readonly value: string;
  readonly options?: CookieOptions;
}

/**
 * Translate Supabase SSR cookie settings into the subset accepted by Next.js.
 *
 * @param cookie Cookie payload emitted by the Supabase SSR client.
 * @returns A cookie object safe to pass to `cookieStore.set`.
 */
function toNextCookie(cookie: SerializableCookie) {
  return {
    domain: cookie.options?.domain,
    expires: cookie.options?.expires,
    httpOnly: cookie.options?.httpOnly,
    maxAge: cookie.options?.maxAge,
    name: cookie.name,
    path: cookie.options?.path,
    sameSite: cookie.options?.sameSite,
    secure: cookie.options?.secure,
    value: cookie.value,
  };
}

/**
 * Read an environment variable that is required for Supabase access.
 *
 * @param name Environment variable name to inspect.
 * @returns The trimmed environment variable value.
 * @throws Error when the variable is missing or blank.
 */
function readRequiredEnvironmentVariable(
  name: "NEXT_PUBLIC_SUPABASE_URL" | "NEXT_PUBLIC_SUPABASE_ANON_KEY",
): string {
  const value = process.env[name]?.trim();

  if (!value) {
    throw new Error(`Missing required environment variable: ${name}.`);
  }

  return value;
}

/**
 * Resolve and validate the shared Supabase environment values.
 *
 * @returns Canonical browser-safe Supabase configuration.
 * @throws Error when the configured URL is invalid.
 */
function resolveSupabaseEnvironment(): SupabaseEnvironment {
  const url = readRequiredEnvironmentVariable("NEXT_PUBLIC_SUPABASE_URL");
  const anonKey = readRequiredEnvironmentVariable("NEXT_PUBLIC_SUPABASE_ANON_KEY");

  try {
    new URL(url);
  } catch (error) {
    throw new Error(
      `Invalid NEXT_PUBLIC_SUPABASE_URL value '${url}'. Expected a valid absolute URL.`,
      { cause: error },
    );
  }

  return { url, anonKey };
}

/**
 * Create a browser-side Supabase client for App Router client components.
 *
 * @returns A configured Supabase browser client using the anon key.
 */
export function createBrowserSupabaseClient() {
  const { url, anonKey } = resolveSupabaseEnvironment();

  return createBrowserClient<Database>(url, anonKey);
}

/**
 * Create a server-side Supabase client bound to the current request cookies.
 *
 * @returns A configured Supabase server client using the anon key.
 */
export async function createServerSupabaseClient() {
  const { url, anonKey } = resolveSupabaseEnvironment();
  const cookieStore = await cookies();

  return createServerClient<Database>(url, anonKey, {
    cookies: {
      getAll() {
        return cookieStore.getAll();
      },
      setAll(cookiesToSet: SerializableCookie[]) {
        for (const cookie of cookiesToSet) {
          cookieStore.set(toNextCookie(cookie));
        }
      },
    },
  });
}
