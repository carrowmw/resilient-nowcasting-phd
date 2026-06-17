// Central resolver for paginated-payload URLs.
//
// Phase 3 (database strategy): payloads are served from Cloudflare R2, NOT from
// local /public static files. The base URL is the bucket's public URL
// (R2_PUBLIC_URL); set NEXT_PUBLIC_PAYLOAD_BASE_URL to override per environment
// (e.g. a custom domain on Vercel). Objects are stored gzip-compressed with
// Content-Encoding: gzip, so fetch() decompresses them transparently.

const RAW_BASE =
  process.env.NEXT_PUBLIC_PAYLOAD_BASE_URL ||
  "https://pub-b102fa91750348f78a6385248654688a.r2.dev";

/** Base URL with any trailing slash stripped. */
export const PAYLOAD_BASE = RAW_BASE.replace(/\/+$/, "");

/** Key prefix under the bucket where the run tree lives. */
const PREFIX = "runs";

export const registryUrl = (): string =>
  `${PAYLOAD_BASE}/${PREFIX}/run_registry.json`;

export const spatialUrl = (runId: string): string =>
  `${PAYLOAD_BASE}/${PREFIX}/${runId}/spatial.json`;

export const pageUrl = (runId: string, date: string): string =>
  `${PAYLOAD_BASE}/${PREFIX}/${runId}/${date}/page.json`;

export const nodeUrl = (runId: string, date: string, nodeId: number): string =>
  `${PAYLOAD_BASE}/${PREFIX}/${runId}/${date}/nodes/node_${nodeId}.json`;
