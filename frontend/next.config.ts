import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    const R2_PUBLIC_URL =
      process.env.R2_PUBLIC_URL ||
      "https://pub-b102fa91750348f78a6385248654688a.r2.dev";

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
        // Immutable cache for historical payloads (spatial.json, page.json, node_*.json)
        source: "/data/runs/:runId/:date*/(.*).(json)",
        headers: [
          {
            key: "Cache-Control",
            value: "public, max-age=31536000, immutable",
          },
        ],
      },
      {
        // Short-lived cache for run_registry.json
        source: "/data/runs/run_registry.json",
        headers: [
          {
            key: "Cache-Control",
            value: "public, s-maxage=60, stale-while-revalidate=30",
          },
        ],
      },
      {
        // Immutable cache for spatial.json (run-level, no date)
        source: "/data/runs/:runId/spatial.json",
        headers: [
          {
            key: "Cache-Control",
            value: "public, max-age=31536000, immutable",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
