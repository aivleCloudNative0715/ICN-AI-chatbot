// src/app/admin/page.tsx
'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import AdminAccessDenied from '@/components/admin/AdminAccessDenied';
import AdminDashboardMain from '@/components/admin/AdminDashboardMain'; // AdminDashboardMain 임포트

export default function AdminPage() {
  const router = useRouter();
  const [isAdmin, setIsAdmin] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // 실제 애플리케이션에서는 백엔드와 토큰을 검증해야 합니다.
    // 현재는 임시 로그인 시 설정된 'user_role' localStorage 값을 사용합니다.
    const userRole = localStorage.getItem('user_role');
    if (userRole === 'ADMIN' || userRole === 'SUPER') {
      setIsAdmin(true);
    } else {
      setIsAdmin(false);
    }
    setLoading(false);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-100">
        <p>인증 확인 중...</p>
      </div>
    );
  }

  if (!isAdmin) {
    return <AdminAccessDenied />;
  }

  // 관리자 권한이 있으면 AdminDashboardMain을 렌더링
  return <AdminDashboardMain />;
}
