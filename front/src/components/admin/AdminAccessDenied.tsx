// src/components/admin/AdminAccessDenied.tsx
'use client';

import { useRouter } from 'next/navigation';

export default function AdminAccessDenied() {
  const router = useRouter();

  const handleGoHome = () => {
    router.push('/');
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <h1 className="text-4xl font-bold text-red-600 mb-8">접근 권한 없음</h1>
      <p className="text-lg text-gray-700 mb-4">
        여기는 관리자만 접근할 수 있는 페이지입니다.
      </p>
      <button
        onClick={handleGoHome}
        className="px-6 py-3 bg-blue-500 text-white rounded-md text-lg hover:bg-blue-600 transition duration-300"
      >
        홈으로 돌아가기
      </button>
    </div>
  );
}