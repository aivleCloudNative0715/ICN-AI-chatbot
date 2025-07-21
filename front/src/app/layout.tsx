// src/app/layout.tsx
'use client';

import './globals.css';
import { Inter } from 'next/font/google';
import Header from '@/components/common/Header';
import AuthModal from '@/components/auth/AuthModal';
import ChatSidebar from '@/components/chat/ChatSidebar'; // ChatSidebar 임포트
import { useState } from 'react';

const inter = Inter({ subsets: ['latin'] });

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false); // 사이드바 상태를 layout으로 이동

  const openAuthModal = () => setIsAuthModalOpen(true);
  const closeAuthModal = () => {
    setIsAuthModalOpen(false);
    // 실제 로그인/회원가입 성공 시 isLoggedIn 상태 업데이트 로직은
    // AuthModal 내부에서 처리하는 것이 더 적합할 수 있습니다.
    // 여기서는 단순히 모달을 닫는 역할만.
  };
  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  // 임시 로그아웃 핸들러
  const handleLogout = () => {
    setIsLoggedIn(false);
    alert('로그아웃되었습니다.');
    // TODO: 실제 API 호출 및 토큰 삭제 로직 추가 (API-08-20030)
  };

  return (
    <html lang="ko">
      <body className={`${inter.className} min-h-screen flex flex-col bg-blue-50`}>
        <Header
          onLoginClick={openAuthModal}
          onRegisterClick={openAuthModal}
          onMenuClick={toggleSidebar} // Header에 사이드바 토글 함수 전달
          isLoggedIn={isLoggedIn}
          onLogoutClick={handleLogout}
        />

        <main className="flex-grow flex items-center justify-center pt-16 bg-blue-50">
          {children}
        </main>

        {/* 사이드바 (조건부 렌더링) */}
        {isSidebarOpen && (
          <ChatSidebar isLoggedIn={isLoggedIn} onClose={toggleSidebar} /> // 사이드바 닫기 함수 전달
        )}

        {/* 인증 모달 */}
        {isAuthModalOpen && (
          <AuthModal
            onClose={() => {
              closeAuthModal();
              // 모달 닫힐 때 로그인 상태 강제 변경 (테스트용)
              // 실제로는 로그인 성공 API 응답에서 isLoggedIn을 true로 설정해야 함
              setIsLoggedIn(true); // 로그인/회원가입 성공 시 이 곳에서 상태 업데이트
            }}
          />
        )}
      </body>
    </html>
  );
}