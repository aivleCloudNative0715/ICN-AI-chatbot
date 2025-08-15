// src/app/providers.tsx
'use client';

import { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider } from '@/contexts/AuthContext';

export default function Providers({ children }: { children: React.ReactNode }) {
  // QueryClient 인스턴스를 컴포넌트 외부나 useState를 이용해 한번만 생성되도록 합니다.
  const [queryClient] = useState(() => new QueryClient());

  return (
    // QueryClientProvider로 전체 앱을 감싸줍니다.
    <QueryClientProvider client={queryClient}>
      {/* 기존에 사용하시던 AuthProvider도 이곳에서 함께 관리합니다. */}
      <AuthProvider>
        {children}
      </AuthProvider>
    </QueryClientProvider>
  );
}