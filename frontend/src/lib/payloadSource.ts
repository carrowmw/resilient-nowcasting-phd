// Central resolver for paginated-payload URLs.
//
// Phase 3 (CDN migration): payloads are proxied through Next.js rewrites
// (/data/runs/* → R2_PUBLIC_URL/runs/*), so all URLs are relative to the
// frontend origin. This avoids CORS, enables Vercel edge caching, and
// centralizes cache-control headers in next.config.ts.

/** Key prefix under the bucket where the run tree lives. */
const PREFIX = "data/runs";

export const registryUrl = (): string => `/${PREFIX}/run_registry.json`;

export const spatialUrl = (runId: string): string =>
  `/${PREFIX}/${runId}/spatial.json`;

export const pageUrl = (runId: string, date: string): string =>
  `/${PREFIX}/${runId}/${date}/page.json`;

export const nodeUrl = (runId: string, date: string, nodeId: number): string =>
  `/${PREFIX}/${runId}/${date}/nodes/node_${nodeId}.json`;
