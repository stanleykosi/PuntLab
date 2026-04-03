/**
 * Bootstrap entry point for the PuntLab Fastify API.
 *
 * Purpose: create and start the HTTP server that will later expose
 * recommendation, history, subscription, and admin routes.
 * Scope: environment parsing, Fastify plugin registration, baseline route
 * wiring, and structured startup/error handling.
 * Dependencies: Fastify, `@fastify/cors`, `@fastify/cookie`, and `dotenv`.
 */

import cookie from "@fastify/cookie";
import cors from "@fastify/cors";
import Fastify, { type FastifyInstance, type FastifyServerOptions } from "fastify";
import dotenv from "dotenv";
import { fileURLToPath } from "node:url";

dotenv.config();

/**
 * Runtime configuration required to start the PuntLab API service.
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
   * Shared database URL; required for later Prisma usage.
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

/**
 * Parse and validate a port number from an environment variable.
 *
 * @param value Raw port string from the environment.
 * @param fallback Canonical default port when no value is provided.
 * @returns A validated TCP port number.
 * @throws Error when the supplied value is not a valid integer port.
 */
function parsePort(value: string | undefined, fallback: number): number {
  if (value === undefined || value.trim() === "") {
    return fallback;
  }

  const parsed = Number(value);

  if (!Number.isInteger(parsed) || parsed < 1 || parsed > 65535) {
    throw new Error(`Invalid PORT value '${value}'. Expected an integer between 1 and 65535.`);
  }

  return parsed;
}

/**
 * Normalize and validate the configured environment name.
 *
 * @param value Raw `ENVIRONMENT` value.
 * @returns A supported deployment environment string.
 * @throws Error when the value is outside the supported environment set.
 */
function parseEnvironment(value: string | undefined): ApiConfig["environment"] {
  const normalized = (value ?? "development").trim().toLowerCase();

  if (
    normalized !== "development" &&
    normalized !== "staging" &&
    normalized !== "production"
  ) {
    throw new Error(
      `Invalid ENVIRONMENT value '${value ?? ""}'. Expected development, staging, or production.`,
    );
  }

  return normalized;
}

/**
 * Normalize and validate the configured log level.
 *
 * @param value Raw `LOG_LEVEL` value.
 * @returns A supported Fastify log level.
 * @throws Error when the value is outside the supported log level set.
 */
function parseLogLevel(value: string | undefined): ApiConfig["logLevel"] {
  const normalized = (value ?? "info").trim().toLowerCase();

  if (
    normalized !== "debug" &&
    normalized !== "info" &&
    normalized !== "warn" &&
    normalized !== "error"
  ) {
    throw new Error(
      `Invalid LOG_LEVEL value '${value ?? ""}'. Expected debug, info, warn, or error.`,
    );
  }

  return normalized;
}

/**
 * Build the API configuration from process environment variables.
 *
 * @returns A validated runtime configuration object for server startup.
 */
export function resolveConfig(): ApiConfig {
  return {
    appName: "puntlab-api",
    environment: parseEnvironment(process.env.ENVIRONMENT),
    port: parsePort(process.env.PORT, DEFAULT_PORT),
    host: process.env.HOST?.trim() || DEFAULT_HOST,
    publicApiUrl: process.env.NEXT_PUBLIC_API_URL?.trim() || DEFAULT_PUBLIC_API_URL,
    databaseUrl: process.env.DATABASE_URL?.trim() || DEFAULT_DATABASE_URL,
    redisUrl: process.env.REDIS_URL?.trim() || DEFAULT_REDIS_URL,
    supabaseUrl: process.env.SUPABASE_URL?.trim() || "",
    supabaseAnonKey: process.env.SUPABASE_ANON_KEY?.trim() || "",
    supabaseServiceKey: process.env.SUPABASE_SERVICE_KEY?.trim() || "",
    paystackSecretKey: process.env.PAYSTACK_SECRET_KEY?.trim() || "",
    paystackPublicKey: process.env.PAYSTACK_PUBLIC_KEY?.trim() || "",
    paystackWebhookSecret: process.env.PAYSTACK_WEBHOOK_SECRET?.trim() || "",
    logLevel: parseLogLevel(process.env.LOG_LEVEL),
  };
}

/**
 * Register the core Fastify plugins used by the API.
 *
 * @param app The Fastify instance being prepared for runtime use.
 * @param config Validated runtime configuration.
 */
async function registerPlugins(app: FastifyInstance, config: ApiConfig): Promise<void> {
  await app.register(cors, {
    origin: [config.publicApiUrl],
    credentials: true,
  });

  await app.register(cookie, {
    hook: "onRequest",
  });
}

/**
 * Register baseline routes needed for bootstrap verification.
 *
 * @param app The Fastify instance receiving route definitions.
 * @param config Validated runtime configuration.
 */
function registerRoutes(app: FastifyInstance, config: ApiConfig): void {
  app.get("/health", async () => {
    return {
      status: "success",
      data: {
        service: config.appName,
        environment: config.environment,
        uptimeSeconds: Math.round(process.uptime()),
      },
    };
  });

  app.get("/api/v1", async () => {
    return {
      status: "success",
      data: {
        service: config.appName,
        message: "PuntLab API bootstrap is online.",
        routesPlanned: [
          "/api/v1/today",
          "/api/v1/history",
          "/api/v1/stats",
          "/api/v1/subscribe",
          "/api/v1/webhooks/paystack",
          "/api/v1/admin/runs",
        ],
      },
    };
  });
}

/**
 * Create a configured Fastify instance without binding a network port.
 *
 * @param overrides Optional Fastify server overrides for tests or tooling.
 * @returns A fully registered Fastify application instance.
 */
export async function createServer(
  overrides: FastifyServerOptions = {},
): Promise<FastifyInstance> {
  const config = resolveConfig();
  const app = Fastify({
    logger: {
      level: config.logLevel,
    },
    ...overrides,
  });

  app.setErrorHandler((error, request, reply) => {
    const message = error instanceof Error ? error.message : "Internal Server Error";

    request.log.error({ err: error }, "Unhandled API error");
    void reply.status(500).send({
      status: "error",
      error: "Internal Server Error",
      message:
        config.environment === "production"
          ? "The API could not process the request."
          : message,
    });
  });

  await registerPlugins(app, config);
  registerRoutes(app, config);

  return app;
}
/**
 * Start the Fastify server and bind the configured host and port.
 *
 * @returns A running Fastify instance.
 */
export async function startServer(): Promise<FastifyInstance> {
  const config = resolveConfig();
  const app = await createServer();

  await app.listen({
    host: config.host,
    port: config.port,
  });

  app.log.info(
    {
      host: config.host,
      port: config.port,
      environment: config.environment,
    },
    "PuntLab API server started",
  );

  return app;
}

const isEntryModule =
  process.argv[1] !== undefined && fileURLToPath(import.meta.url) === process.argv[1];

if (isEntryModule) {
  startServer().catch((error: unknown) => {
    const message = error instanceof Error ? error.message : "Unknown startup error";
    console.error(`Failed to start PuntLab API: ${message}`);
    process.exitCode = 1;
  });
}
