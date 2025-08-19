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

  async rewrites() {
    return [
      // 1. 일반 API 요청을 위한 프록시 규칙 (HTTPS 서버로)
      {
        // source: 이 경로로 들어오는 요청을 감지합니다.
        source: "/api/:path*",
        // destination: 이 주소로 요청을 대신 보냅니다.
        destination: `${process.env.NEXT_PUBLIC_API_BASE_URL}/api/:path*`,
      },

      // 2. 파일 업로드를 위한 AI 서버 프록시 규칙 (HTTP 서버로)
      {
        source: "/api-ai/:path*",
        destination: `${process.env.NEXT_PUBLIC_AI_SERVER_URL}/:path*`,
      },
    ];
  },
};

export default nextConfig;
