import type { NextConfig } from "next";
import { fileURLToPath } from "node:url";

const platformApi = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/$/, "");
const frontendRoot = fileURLToPath(new URL(".", import.meta.url));

const nextConfig: NextConfig = {
  // Standalone exists for `nomicous/frontend/Dockerfile`, which copies
  // `.next/standalone`. Vercel traces the server itself and its onBuildComplete
  // hook reads `.next/next-server.js.nft.json`, which the Turbopack standalone
  // build does not emit as of Next 16.3.0 - the deploy fails on ENOENT. Leave
  // `output` unset there so Vercel takes its own supported path.
  output: process.env.VERCEL ? undefined : "standalone",
  // Keep legacy route components in src/pages while App Router owns routing.
  pageExtensions: ["next.tsx", "next.ts", "next.jsx", "next.js"],
  turbopack: {
    root: frontendRoot,
  },
  async rewrites() {
    return [
      {
        source: "/media/:path*",
        destination: `${platformApi}/media/:path*`,
      },
    ];
  },
};

export default nextConfig;
