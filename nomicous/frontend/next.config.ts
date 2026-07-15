import type { NextConfig } from "next";
import { fileURLToPath } from "node:url";

const platformApi = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/$/, "");
const frontendRoot = fileURLToPath(new URL(".", import.meta.url));
const helperOrigins = "http://localhost:8001 http://127.0.0.1:8001";

function platformApiOrigin(): string | null {
  try {
    return new URL(platformApi).origin;
  } catch {
    return null;
  }
}

function buildContentSecurityPolicy(): string {
  const apiOrigin = platformApiOrigin();
  const connect = ["'self'", helperOrigins];
  const img = ["'self'", "data:", "blob:"];
  if (apiOrigin) {
    connect.push(apiOrigin);
    img.push(apiOrigin);
  }
  return [
    "default-src 'self'",
    "base-uri 'self'",
    "object-src 'none'",
    "frame-ancestors 'none'",
    "form-action 'self'",
    "script-src 'self' 'unsafe-inline'",
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com",
    `img-src ${img.join(" ")}`,
    `connect-src ${connect.join(" ")}`,
    "worker-src 'self' blob:",
  ].join("; ");
}

const nextConfig: NextConfig = {
  output: "standalone",
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
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          {
            key: "Strict-Transport-Security",
            value: "max-age=31536000; includeSubDomains; preload",
          },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          {
            key: "Referrer-Policy",
            value: "strict-origin-when-cross-origin",
          },
          {
            key: "Permissions-Policy",
            value:
              "camera=(), geolocation=(), microphone=(), payment=(), usb=()",
          },
          {
            key: "Content-Security-Policy",
            value: buildContentSecurityPolicy(),
          },
        ],
      },
    ];
  },
};

export default nextConfig;
