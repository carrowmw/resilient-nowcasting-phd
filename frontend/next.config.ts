import type { NextConfig } from "next";

// Short-lived for the registry so new runs appear quickly, but allow stale reads during revalidation.
const REGISTRY_MAX_AGE = 60;
const REGISTRY_SWR = 30;

const nextConfig: NextConfig = {
  async rewrites() {
    const R2_PUBLIC_URL = process.env.R2_PUBLIC_URL;
    if (!R2_PUBLIC_URL) {
      throw new Error(
        "FATAL: R2_PUBLIC_URL environment variable is not set. " +
          "Add it to .env.local or the deployment environment. See .env.example."
      );
    }

    return [
      {
        source: "/data/runs/:path*",
        destination: `${R2_PUBLIC_URL}/runs/:path*`,
      },
    ];
  },

  async headers() {
    return [
      {
        // Mutable cache for payloads to allow re-inference updates
        source: "/data/runs/:runId/:date*/(.*).(json)",
        headers: [
          {
            key: "Cache-Control",
            value: "public, s-maxage=3600, stale-while-revalidate=86400",
          },
        ],
      },
      {
        // Short-lived cache for run_registry.json
        source: "/data/runs/run_registry.json",
        headers: [
          {
            key: "Cache-Control",
            value: `public, s-maxage=${REGISTRY_MAX_AGE}, stale-while-revalidate=${REGISTRY_SWR}`,
          },
        ],
      },
      {
        // Mutable cache for spatial.json (run-level, no date) to allow re-inference updates
        source: "/data/runs/:runId/spatial.json",
        headers: [
          {
            key: "Cache-Control",
            value: "public, s-maxage=3600, stale-while-revalidate=86400",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
