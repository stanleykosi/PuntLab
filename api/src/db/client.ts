/**
 * Direct PostgreSQL client utilities for the PuntLab API.
 *
 * Purpose: provide a lightweight SQL-first database integration for Fastify
 * without introducing a second schema source of truth.
 * Scope: client creation, Fastify decoration, lifecycle management, and a
 * minimal validated health check query.
 * Dependencies: `postgres` for the database connection and `zod` for row
 * validation of infrastructure queries.
 */

import type { FastifyInstance } from "fastify";
import postgres, { type Sql } from "postgres";
import { z } from "zod";

/**
 * Postgres client shape shared across the API codebase.
 */
export type DatabaseClient = Sql<Record<string, unknown>>;

declare module "fastify" {
  interface FastifyInstance {
    /**
     * Shared PostgreSQL client for raw SQL queries.
     */
    db: DatabaseClient;
  }
}

const databaseHealthRowSchema = z.object({
  result: z.number().int(),
});

const databaseHealthRowsSchema = z.array(databaseHealthRowSchema).min(1);

/**
 * Create a reusable PostgreSQL client using the canonical database URL.
 *
 * @param databaseUrl PostgreSQL connection string for the target database.
 * @returns A configured `postgres` SQL client instance.
 */
export function createDatabaseClient(databaseUrl: string): DatabaseClient {
  return postgres(databaseUrl, {
    max: 10,
    connect_timeout: 10,
    idle_timeout: 20,
    prepare: false,
  });
}

/**
 * Attach the shared database client to the Fastify instance and ensure it
 * closes cleanly during server shutdown.
 *
 * @param app Fastify application receiving the database decoration.
 * @param databaseUrl PostgreSQL connection string for the API runtime.
 */
export function registerDatabase(app: FastifyInstance, databaseUrl: string): void {
  const client = createDatabaseClient(databaseUrl);

  app.decorate("db", client);

  app.addHook("onClose", async (instance) => {
    await instance.db.end({ timeout: 5 });
  });
}

/**
 * Run a simple validated query to confirm the database connection is usable.
 *
 * @param app Fastify application with an attached database client.
 * @returns A small health payload confirming query success.
 */
export async function checkDatabaseHealth(
  app: FastifyInstance,
): Promise<{ readonly connected: true }> {
  const rows = databaseHealthRowsSchema.parse(
    await app.db<{ result: number }[]>`SELECT 1 AS result`,
  );
  const row = rows[0];

  if (row === undefined || row.result !== 1) {
    throw new Error("Database health check returned an unexpected result.");
  }

  return { connected: true };
}
