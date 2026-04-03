/**
 * Bootstrap entry point for the PuntLab Fastify API.
 *
 * Purpose: create and start the HTTP server that will later expose
 * recommendation, history, subscription, and admin routes.
 * Scope: environment parsing, Fastify plugin registration, baseline route
 * wiring, and structured startup/error handling.
 * Dependencies: Fastify, `@fastify/cors`, `@fastify/cookie`, and the
 * SQL-first infrastructure modules in `src/config.ts` and `src/db/client.ts`.
 */

import cookie from "@fastify/cookie";
import cors from "@fastify/cors";
import Fastify, { type FastifyInstance, type FastifyServerOptions } from "fastify";
import { fileURLToPath } from "node:url";
import { type ApiConfig, resolveConfig } from "./config.js";
import { checkDatabaseHealth, registerDatabase } from "./db/client.js";

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

  registerDatabase(app, config.databaseUrl);
}

/**
 * Register baseline routes needed for bootstrap verification.
 *
 * @param app The Fastify instance receiving route definitions.
 * @param config Validated runtime configuration.
 */
function registerRoutes(app: FastifyInstance, config: ApiConfig): void {
  app.get("/health", async (_, reply) => {
    try {
      await checkDatabaseHealth(app);

      return {
        status: "success",
        data: {
          service: config.appName,
          environment: config.environment,
          uptimeSeconds: Math.round(process.uptime()),
          database: "connected",
        },
      };
    } catch (error) {
      reply.code(503);

      return {
        status: "error",
        data: {
          service: config.appName,
          environment: config.environment,
          uptimeSeconds: Math.round(process.uptime()),
          database: "unavailable",
        },
        error: {
          message: error instanceof Error ? error.message : "Unknown database error.",
        },
      };
    }
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
