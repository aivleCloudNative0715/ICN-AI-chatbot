// tailwind.config.js (프로젝트 루트에 위치)
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './node_modules/primereact/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // 메인 브랜드 색상
        'primary': '#003F8F', // 진한 파랑

        // 보조 파랑 계열 (경량화)
        'secondary': {
          DEFAULT: '#ABC0DA', // 회색빛 파랑 (기존 ABC0DA)
          light: '#EFF6FF',   // 아주 연한 파랑 (기존 EFF6FF)
          dark: '#507098',    // 차분한 파랑/회색 (기존 507098)
        },

        // 게시판 색상
        'board': {
          primary: '#DBF0F3',
          light: '#C2E0E3',
          dark: '#0C5D66'
        },

        // 강조 노란색
        'accent-yellow': '#F2B705', // 강조 노란색

        // 중립 색상 (회색, 검정, 흰색)
        'neutral': {
          'white': '#FFFFFF',    // 흰색
          'black': '#000000',    // 검정색
          'dark-text': '#0F172A',// 아주 진한 회색 (거의 검정), 기본 텍스트에 적합
          'medium-gray': '#A3A2A2', // 중간 회색
        },

        // 시스템/특정 목적 색상 (예: 구글 버튼)
        'google-blue': '#4285F4', // 구글 로그인용 파랑
        'error-red': '#EA4335',   // 오류 메시지 등 빨강
        'success-green': '#34A853', // 성공 메시지 등 초록
      },
    },
  },
  plugins: [],
};