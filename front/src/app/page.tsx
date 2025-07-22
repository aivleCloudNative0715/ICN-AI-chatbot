// src/app/page.tsx
'use client';

import ChatBotScreen from '@/components/chat/ChatBotScreen';
import ChatSidebar from '@/components/chat/ChatSidebar';
import Header from '@/components/common/Header'; // Header 컴포넌트 임포트
import AuthModal from '@/components/auth/AuthModal'; // AuthModal 컴포넌트 임포트
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function HomePage() {
  const router = useRouter();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');

  useEffect(() => {
    const token = localStorage.getItem('jwt_token');
    if (token) {
      // TODO: 토큰 유효성 검사 API 호출
      setIsLoggedIn(true);
    }
  }, []);

  const openAuthModal = (mode: 'login' | 'register' = 'login') => {
    setAuthMode(mode);
    setIsAuthModalOpen(true);
  };
  
  const closeAuthModal = () => {
    // 창이 닫히면 로그인 되도록 임시 처리함 (나중에 지워야함)
    setIsLoggedIn(true);
    setIsAuthModalOpen(false);
  };

  const handleLoginSuccess = () => {
    setIsLoggedIn(true);
    closeAuthModal();
  }

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  const handleLogout = () => {
    // TODO: 실제 API 호출 및 토큰 삭제 로직 (API-08-20030)
    localStorage.removeItem('jwt_token');
    setIsLoggedIn(false);
    alert('로그아웃되었습니다.');
    // 현재 챗봇 페이지이므로 특별한 라우팅 필요 없음
  };

  return (
    <div className="flex flex-col flex-1 w-full h-full"> {/* 전체 화면 차지 */}
      {/* 챗봇용 헤더 */}
      <Header
        onLoginClick={() => openAuthModal('login')}
        onRegisterClick={() => openAuthModal('register')}
        onMenuClick={toggleSidebar}
        isLoggedIn={isLoggedIn}
        onLogoutClick={handleLogout}
      />

      {/* 메인 콘텐츠 영역 (챗봇 화면) */}
      <div className="flex-grow flex overflow-hidden">
        <ChatBotScreen
          isLoggedIn={isLoggedIn}
          onLoginStatusChange={setIsLoggedIn} // 로그인 상태 변경 직접 전달
          onSidebarToggle={toggleSidebar} // 사이드바 토글 핸들러 전달
        />
      </div>
      

      {/* 챗봇 사이드바 (조건부 렌더링) */}
      {isSidebarOpen && (
        <ChatSidebar isLoggedIn={isLoggedIn} onClose={toggleSidebar} />
      )}

      {/* 인증 모달 */}
      {isAuthModalOpen && (
        <AuthModal
          onClose={closeAuthModal}
          onLoginSuccess={handleLoginSuccess}
          initialMode={authMode}
        />
      )}
    </div>
  );
}