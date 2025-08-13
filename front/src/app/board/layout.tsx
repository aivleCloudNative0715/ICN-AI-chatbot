// src/app/board/layout.tsx
'use client';

import React, { useEffect } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import FloatingActionButton from '@/components/common/FloatingActionButton';
import BoardHeader from '@/components/board/BoardHeader';
import { useAuth } from '@/contexts/AuthContext'; // AuthContext 가져오기

interface BoardLayoutProps {
  children: React.ReactNode;
}

export default function BoardLayout({ children }: BoardLayoutProps) {
  const router = useRouter();
  const pathname = usePathname();
  
  // Context에서 실제 로그인 정보와 로그아웃 함수를 가져온다
  const { isLoggedIn, logout } = useAuth();

  const handleGoToChat = () => {
    router.push('/');
  };

  const handleNewInquiryClick = () => {
    if (isLoggedIn) {
      router.push('/board/new');
    } else {
      alert('로그인 후 이용 가능합니다.');
      // TODO: 로그인 모달 열기
    }
  };

  useEffect(() => {
    // AuthContext의 초기화 시간을 고려하여 token 유무로 체크
    const storedToken = localStorage.getItem('jwt_token');
    if (!storedToken) {
      alert('로그인이 필요한 서비스입니다.');
      router.replace('/');
    }
  }, [router]);
  
  // 새 글 작성 페이지에서는 플로팅 버튼을 숨깁니다.
  const isNewPage = pathname.startsWith('/board/new');

  return (
    <div className="flex flex-col flex-1 h-screen overflow-hidden bg-board-primary">
      <BoardHeader
        onGoToChat={handleGoToChat}
        onLogout={logout}
      />
      <div className="flex flex-grow overflow-hidden">
        {/* 페이지 콘텐츠 */}
        <main className="flex-grow flex flex-col overflow-y-auto p-6">
          {children}
        </main>
        
        {/* 새 글 작성 페이지가 아닐 때만 플로팅 버튼 표시 */}
        {!isNewPage && (
          <FloatingActionButton onClick={handleNewInquiryClick} />
        )}
      </div>
    </div>
  );
}