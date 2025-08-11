import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  //----------------------------
  // 프론트 완성 후 false로 수정하거나 지워야함
  eslint: {
    // 이 옵션은 ESLint 에러가 있어도 빌드를 성공시킵니다.
    ignoreDuringBuilds: true,
  },
  //-----------------------
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.API_BASE_URL}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
