import type { NextConfig } from 'next';
import { fileURLToPath } from 'node:url';

const platformApi = (process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000').replace(
  /\/$/,
  '',
);
const frontendRoot = fileURLToPath(new URL('.', import.meta.url));

const nextConfig: NextConfig = {
  output: 'standalone',
  // Keep legacy route components in src/pages while App Router owns routing.
  pageExtensions: ['next.tsx', 'next.ts', 'next.jsx', 'next.js'],
  turbopack: {
    root: frontendRoot,
  },
  async rewrites() {
    return [
      {
        source: '/media/:path*',
        destination: `${platformApi}/media/:path*`,
      },
    ];
  },
};

export default nextConfig;
