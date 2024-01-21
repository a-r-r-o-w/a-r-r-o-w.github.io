/** @type {import("next").NextConfig} */
const nextConfig = {
  "output": "export",
  images: {
    unoptimized: true,
  },
  distDir: "docs",
  trailingSlash: true,
};

module.exports = nextConfig;
