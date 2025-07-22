// src/app/board/layout.tsx
'use client';

import React, { useState, useEffect } from 'react';
import { useRouter, usePathname } from 'next/navigation'; // usePathname 임포트 추가
import BoardSidebar from '@/components/board/BoardSidebar';
import FloatingActionButton from '@/components/common/FloatingActionButton';
import BoardHeader from '@/components/board/BoardHeader';

interface BoardLayoutProps {
  children: React.ReactNode;
}

export default function BoardLayout({ children }: BoardLayoutProps) {
  const router = useRouter();
  const pathname = usePathname(); // usePathname 훅 사용
  // TODO: 실제 isLoggedIn 상태는 Context API 또는 전역 상태 관리 훅에서 가져와야 함
  // 현재는 로그인이 되어있다고 가정함
  const [isLoggedIn, setIsLoggedIn] = useState(true); // 게시판 레이아웃에서도 로그인 상태 관리 필요

  useEffect(() => {
    const token = localStorage.getItem('jwt_token');
    if (token) {
      // TODO: 토큰 유효성 검사 API 호출
      setIsLoggedIn(true);
    }
  }, []);

  const handleGoToChat = () => {
    router.push('/'); // 메인 챗봇 화면으로 이동
  };

  const handleLogout = () => {
    // TODO: 실제 API 호출 및 토큰 삭제 로직 (API-08-20030)
    localStorage.removeItem('jwt_token');
    setIsLoggedIn(false);
    alert('로그아웃되었습니다.');
    router.push('/'); // 로그아웃 후 메인 챗봇 화면으로 이동
  };

  const handleNewInquiryClick = () => {
    if (isLoggedIn) {
      router.push('/board/new');
    } else {
      alert('로그인 후 이용 가능합니다.');
      // TODO: 로그인 모달 열기 등의 액션
    }
  };

  const isNewPage = pathname === '/board/new'; // usePathname으로 가져온 pathname 사용

  return (
    <div className="flex flex-col flex-1 h-screen overflow-hidden bg-board-primary">
      {/* 게시판 전용 헤더 */}
      <BoardHeader
        onGoToChat={handleGoToChat}
        onLogout={handleLogout}
      />

      <div className="flex flex-grow"> {/* 사이드바와 메인 콘텐츠를 위한 flex-grow */}
        {/* 사이드바 */}
        <BoardSidebar isLoggedIn={isLoggedIn} />

        {/* 페이지 콘텐츠 */}
        <main className="flex-grow flex flex-col overflow-y-auto p-6">
          {children}
        </main>

        {/* 새 문의/건의 추가 Floating Action Button */}
        {!isNewPage && isLoggedIn && (
          <FloatingActionButton onClick={handleNewInquiryClick} label="새 문의/건의" />
        )}
      </div>
    </div>
  );
}