/**
 * Cloudflare Worker to dynamically rebuild the run_registry.json
 * from the contents of an R2 bucket.
 *
 * Mirrors scripts/build_run_registry.py: RunRegistry (schema_version 5,
 * frontend/src/types/registry.ts), runs ordered by registry.rank, the reference
 * run (v5_full-s42) as the default. spatial.json files that are not schema 5 are
 * skipped, so stale v4 objects can never reach the frontend.
 */

export interface Env {
  PAYLOAD_BUCKET: R2Bucket;
  WORKER_REBUILD_TOKEN: string;
}

const SCHEMA_VERSION = 5;

// RunInfo fields + RunRegistryEntry fields copied verbatim from spatial.json metadata.
const ENTRY_KEYS = [
  "run_id", "experiment", "group", "description", "model_type", "forecast_likelihood",
  "quantile_levels", "capabilities", "seq_len", "wandb_run_id", "wandb_artifact",
  "git_commit_hash", "generated_at",
  "registry", "available_dates", "num_nodes", "num_holdout_nodes", "num_ghost_nodes", "holdout_manifest_sha",
  "test_metrics", "window_metrics", "hyperparameters",
] as const;

async function readJson(bucket: R2Bucket, key: string): Promise<any | null> {
  const file = await bucket.get(key);
  if (!file) return null;
  let bodyStream = file.body;
  if (file.httpMetadata?.contentEncoding === "gzip") {
    // @ts-ignore - TS dom types sometimes lack DecompressionStream but it exists in CF Workers
    bodyStream = bodyStream.pipeThrough(new DecompressionStream("gzip"));
  }
  return new Response(bodyStream).json();
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    if (request.method !== "POST") {
      return new Response("Method not allowed. Use POST.", { status: 405 });
    }

    // Verify Auth
    const authHeader = request.headers.get("Authorization");
    if (!authHeader || authHeader !== `Bearer ${env.WORKER_REBUILD_TOKEN}`) {
      return new Response("Unauthorized", { status: 401 });
    }

    const url = new URL(request.url);
    if (url.pathname !== "/rebuild") {
      return new Response("Not found", { status: 404 });
    }

    try {
      const runs: any[] = [];
      const skipped: string[] = [];
      let cursor: string | undefined = undefined;

      // One "directory" per run: list runs/<run_id>/ prefixes, not every day object.
      do {
        const listed: R2Objects = await env.PAYLOAD_BUCKET.list({
          prefix: "runs/",
          delimiter: "/",
          cursor: cursor,
        });

        for (const prefix of listed.delimitedPrefixes) {
          const key = `${prefix}spatial.json`;
          const data = await readJson(env.PAYLOAD_BUCKET, key);
          const metadata = data?.metadata;
          if (!metadata || metadata.schema_version !== SCHEMA_VERSION) {
            console.warn(`Skipping ${key}: missing or not schema_version ${SCHEMA_VERSION}`);
            skipped.push(key);
            continue;
          }
          const missing = ENTRY_KEYS.filter((k) => !(k in metadata));
          if (missing.length > 0) {
            console.warn(`Skipping ${key}: metadata missing ${missing.join(", ")}`);
            skipped.push(key);
            continue;
          }

          const entry: Record<string, unknown> = {};
          for (const k of ENTRY_KEYS) entry[k] = metadata[k];
          entry.analysis_period = { start_date: metadata.start_date, end_date: metadata.end_date };
          runs.push(entry);
        }

        cursor = listed.truncated ? listed.cursor : undefined;
      } while (cursor);

      runs.sort((a, b) => a.registry.rank - b.registry.rank);
      const reference = runs.find((r) => r.registry.is_reference);
      const rules = new Set(runs.map((r) => r.registry.selection_rule));

      const registry = {
        schema_version: SCHEMA_VERSION,
        generated_at: new Date().toISOString(),
        default_run_id: reference?.run_id ?? runs[0]?.run_id ?? null,
        selection_rule: rules.size === 1 ? [...rules][0] : "mixed",
        runs: runs,
      };

      // Write back to R2
      await env.PAYLOAD_BUCKET.put("runs/run_registry.json", JSON.stringify(registry), {
        httpMetadata: {
          contentType: "application/json",
          cacheControl: "public, max-age=60",
        },
      });

      return new Response(JSON.stringify({ success: true, count: runs.length, skipped }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });

    } catch (error: any) {
      console.error(error);
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      });
    }
  },
};
