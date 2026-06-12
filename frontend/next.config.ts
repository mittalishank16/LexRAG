// frontend/next.config.ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",   // ← add this line — enables the Dockerfile.frontend above
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "*.onrender.com" },
    ],
  },
};

export default nextConfig;