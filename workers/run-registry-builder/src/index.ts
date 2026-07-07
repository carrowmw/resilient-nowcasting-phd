/**
 * Cloudflare Worker to dynamically rebuild the run_registry.json
 * from the contents of an R2 bucket.
 */

export interface Env {
  PAYLOAD_BUCKET: R2Bucket;
  WORKER_REBUILD_TOKEN: string;
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
      const runs = [];
      let defaultRunId: string | null = null;
      let cursor: string | undefined = undefined;

      // Paginate through all runs/*/spatial.json keys
      do {
        const listed = await env.PAYLOAD_BUCKET.list({
          prefix: "runs/",
          cursor: cursor,
        });

        const spatialKeys = listed.objects.filter((obj) => obj.key.endsWith("/spatial.json"));

        for (const obj of spatialKeys) {
          const file = await env.PAYLOAD_BUCKET.get(obj.key);
          if (!file) continue;

          const encoding = file.httpMetadata?.contentEncoding;
          let bodyStream = file.body;
          if (encoding === "gzip") {
            // @ts-ignore - TS dom types sometimes lack DecompressionStream but it exists in CF Workers
            bodyStream = bodyStream.pipeThrough(new DecompressionStream("gzip"));
          }

          const response = new Response(bodyStream);
          const data = await response.json();
          const metadata = data.metadata;
          
          if (!metadata || !metadata.run_id) {
            console.warn(`Skipping ${obj.key}: Invalid metadata format`);
            continue;
          }

          runs.push({
            run_id: metadata.run_id,
            wandb_run_id: metadata.wandb_run_id,
            wandb_run_name: metadata.run_id, // we use run_id for wandb_run_name as fallback in python
            git_commit: metadata.git_commit_hash,
            generated_at: metadata.generated_at,
            analysis_period: {
              start_date: metadata.start_date,
              end_date: metadata.end_date,
            },
            train_config: metadata.hyperparameters || {},
            metrics_summary: metadata.metrics_summary || {},
            available_dates: metadata.available_dates || [metadata.start_date],
            num_nodes: metadata.num_nodes,
            num_simulated_nodes: metadata.num_simulated_nodes,
          });
        }
        
        cursor = listed.truncated ? listed.cursor : undefined;
      } while (cursor);
      
      // Sort runs by generated_at descending (newest first)
      runs.sort((a, b) => new Date(b.generated_at).getTime() - new Date(a.generated_at).getTime());
      
      if (runs.length > 0) {
          defaultRunId = runs[0].run_id;
      }

      const registry = {
        default_run_id: defaultRunId,
        runs: runs,
      };

      const registryJson = JSON.stringify(registry, null, 2);

      // Write back to R2
      await env.PAYLOAD_BUCKET.put("runs/run_registry.json", registryJson, {
        httpMetadata: {
          contentType: "application/json",
          cacheControl: "public, max-age=60",
        },
      });

      return new Response(JSON.stringify({ success: true, count: runs.length }), {
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
