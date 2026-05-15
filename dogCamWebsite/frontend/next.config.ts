/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://backend:8080/api/:path*',
      },
      {
        source: '/socket.io/:path*',
        destination: 'http://backend:8080/socket.io/:path*',
      },
    ];
  },
}

module.exports = nextConfig
