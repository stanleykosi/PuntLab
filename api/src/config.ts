/**
 * Runtime configuration for the PuntLab Fastify API.
 *
 * Purpose: centralize environment loading and validation for the Node.js API.
 * Scope: parses process environment variables into a strongly typed config
 * object used by the server bootstrap and infrastructure clients.
 * Dependencies: `dotenv` for loading `.env` files and `zod` for validation.
 */

import "dotenv/config";
import { z } from "zod";

/**
 * Strongly typed runtime configuration consumed by the API service.
 */
export interface ApiConfig {
  /**
   * Human-readable service name used in diagnostics.
   */
  readonly appName: string;
  /**
   * Deployment environment for log context and future branching.
   */
  readonly environment: "development" | "staging" | "production";
  /**
   * Port to bind the Fastify server to.
   */
  readonly port: number;
  /**
   * Host interface for the Fastify listener.
   */
  readonly host: string;
  /**
   * Base API origin allowed by the CORS policy.
   */
  readonly publicApiUrl: string;
  /**
   * Shared PostgreSQL connection string used by the direct SQL client.
   */
  readonly databaseUrl: string;
  /**
   * Optional Redis URL for cache-backed reads.
   */
  readonly redisUrl: string;
  /**
   * Optional Supabase URL for server integrations.
   */
  readonly supabaseUrl: string;
  /**
   * Optional anon key for client-like Supabase interactions.
   */
  readonly supabaseAnonKey: string;
  /**
   * Optional service key for privileged Supabase server access.
   */
  readonly supabaseServiceKey: string;
  /**
   * Optional Paystack secret for payment setup.
   */
  readonly paystackSecretKey: string;
  /**
   * Optional Paystack public key for checkout creation.
   */
  readonly paystackPublicKey: string;
  /**
   * Optional webhook secret used to verify Paystack webhooks later.
   */
  readonly paystackWebhookSecret: string;
  /**
   * Log level forwarded to Fastify.
   */
  readonly logLevel: "debug" | "info" | "warn" | "error";
}

const DEFAULT_HOST = "0.0.0.0";
const DEFAULT_PORT = 3001;
const DEFAULT_PUBLIC_API_URL = "http://localhost:3001";
const DEFAULT_DATABASE_URL = "postgresql://puntlab:puntlab@localhost:5432/puntlab";
const DEFAULT_REDIS_URL = "redis://localhost:6379/0";

const environmentSchema = z.enum(["development", "staging", "production"]);
const logLevelSchema = z.enum(["debug", "info", "warn", "error"]);

const apiEnvironmentSchema = z.object({
  ENVIRONMENT: environmentSchema.optional(),
  PORT: z.coerce.number().int().min(1).max(65535).optional(),
  HOST: z.string().trim().min(1).optional(),
  NEXT_PUBLIC_API_URL: z.url().optional(),
  DATABASE_URL: z.string().trim().min(1).optional(),
  REDIS_URL: z.string().trim().min(1).optional(),
  SUPABASE_URL: z.string().trim().optional(),
  SUPABASE_ANON_KEY: z.string().trim().optional(),
  SUPABASE_SERVICE_KEY: z.string().trim().optional(),
  PAYSTACK_SECRET_KEY: z.string().trim().optional(),
  PAYSTACK_PUBLIC_KEY: z.string().trim().optional(),
  PAYSTACK_WEBHOOK_SECRET: z.string().trim().optional(),
  LOG_LEVEL: logLevelSchema.optional(),
});

/**
 * Format a Zod validation error into a compact startup diagnostic.
 *
 * @param error Validation failure returned by Zod.
 * @returns A concise error message suitable for server startup logs.
 */
function formatConfigError(error: z.ZodError): string {
  const issues = error.issues.map((issue) => {
    const path = issue.path.length === 0 ? "environment" : issue.path.join(".");
    return `${path}: ${issue.message}`;
  });

  return `Invalid API environment configuration.\n${issues.join("\n")}`;
}

/**
 * Parse and validate the current process environment into runtime config.
 *
 * @returns A validated configuration object for the API runtime.
 * @throws Error when the environment contains invalid values.
 */
export function resolveConfig(): ApiConfig {
  const parsedEnvironment = apiEnvironmentSchema.safeParse(process.env);

  if (!parsedEnvironment.success) {
    throw new Error(formatConfigError(parsedEnvironment.error));
  }

  const environment = parsedEnvironment.data;

  return {
    appName: "puntlab-api",
    environment: environment.ENVIRONMENT ?? "development",
    port: environment.PORT ?? DEFAULT_PORT,
    host: environment.HOST ?? DEFAULT_HOST,
    publicApiUrl: environment.NEXT_PUBLIC_API_URL ?? DEFAULT_PUBLIC_API_URL,
    databaseUrl: environment.DATABASE_URL ?? DEFAULT_DATABASE_URL,
    redisUrl: environment.REDIS_URL ?? DEFAULT_REDIS_URL,
    supabaseUrl: environment.SUPABASE_URL ?? "",
    supabaseAnonKey: environment.SUPABASE_ANON_KEY ?? "",
    supabaseServiceKey: environment.SUPABASE_SERVICE_KEY ?? "",
    paystackSecretKey: environment.PAYSTACK_SECRET_KEY ?? "",
    paystackPublicKey: environment.PAYSTACK_PUBLIC_KEY ?? "",
    paystackWebhookSecret: environment.PAYSTACK_WEBHOOK_SECRET ?? "",
    logLevel: environment.LOG_LEVEL ?? "info",
  };
}
