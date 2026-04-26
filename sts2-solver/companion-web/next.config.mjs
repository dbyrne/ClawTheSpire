/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  images: { unoptimized: true },
  allowedDevOrigins: ["localhost", "127.0.0.1", "100.100.101.1"],
  // Dev: proxy /api/* -> FastAPI on :8765. In production (static export)
  // rewrites are a no-op; the FastAPI server itself serves the API under
  // /api/ alongside the static bundle.
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8765/api/:path*",
      },
    ];
  },
};

export default nextConfig;
