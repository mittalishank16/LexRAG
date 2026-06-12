// next.config.ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow the Render backend URL for image optimisation if needed
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "*.onrender.com" },
    ],
  },
};

export default nextConfig;
