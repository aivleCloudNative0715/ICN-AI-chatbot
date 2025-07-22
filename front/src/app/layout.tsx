// src/app/layout.tsx
'use client';

import './globals.css';
import { Inter } from 'next/font/google';
import Header from '@/components/common/Header';
import AuthModal from '@/components/auth/AuthModal';
import ChatSidebar from '@/components/chat/ChatSidebar';
import { useState, useEffect } from 'react'; // useEffect 추가

const inter = Inter({ subsets: ['latin'] });

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false); // 로그인 상태
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');

  // 실제 앱에서는 JWT 토큰 유효성 검사 등으로 초기 로그인 상태를 설정합니다.
  useEffect(() => {
    // 예시: localStorage에서 토큰 확인
    const token = localStorage.getItem('jwt_token');
    if (token) {
      // TODO: 토큰 유효성 검사 API 호출
      // 유효하다면 setIsLoggedIn(true)
      setIsLoggedIn(true); // 임시로 토큰이 있으면 로그인 상태로 간주
    }
  }, []);


  const openAuthModal = (mode: 'login' | 'register' = 'login') => {
    setAuthMode(mode);
    setIsAuthModalOpen(true);
  };
  
  const closeAuthModal = () => {
    setIsAuthModalOpen(false);
  };

  const handleLoginSuccess = () => {
    setIsLoggedIn(true); // 로그인 성공 시 상태 업데이트
    closeAuthModal(); // 모달 닫기
  }

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  const handleLogout = () => {
    // TODO: 실제 API 호출 및 토큰 삭제 로직 (API-08-20030)
    localStorage.removeItem('jwt_token'); // 예시: 로컬 스토리지에서 토큰 삭제
    setIsLoggedIn(false);
    alert('로그아웃되었습니다.');
    // 현재 페이지가 게시판이면 메인으로 리다이렉트하는 로직 추가 가능
    // useRouter().push('/');
  };

  return (
    <html lang="ko">
      <body className={`${inter.className} h-screen flex flex-col bg-blue-50`}>
        <Header
          onLoginClick={() => openAuthModal('login')}
          onRegisterClick={() => openAuthModal('register')}
          onMenuClick={toggleSidebar}
          isLoggedIn={isLoggedIn}
          onLogoutClick={handleLogout}
        />

        {/* layout.tsx의 main은 이제 children이 BoardLayout을 포함할 수 있도록 변경 */}
        {/* BoardLayout 내부에서 자체적인 flex-grow와 배경색을 가질 것이므로, 여기서는 기본 구성만 */}
        <div className="flex-grow flex"> {/* flex-grow와 flex를 유지하여 자식 요소가 공간을 차지하도록 */}
          {children}
        </div>

        {/* 사이드바 (조건부 렌더링) */}
        {isSidebarOpen && (
          <ChatSidebar isLoggedIn={isLoggedIn} onClose={toggleSidebar} />
        )}

        {/* 인증 모달 */}
        {isAuthModalOpen && (
          <AuthModal
            // onClose={closeAuthModal}
            onClose={() => {
              closeAuthModal();
              setIsLoggedIn(true)
            }}
            initialMode={authMode}
            // AuthModal 내부에 onLoginSuccess, onRegisterSuccess prop을 전달하여 isLoggedIn 업데이트
            // AuthModal 내에서 LoginForm/RegisterForm의 onLoginSuccess, onRegisterSuccess를 호출
          />
        )}
      </body>
    </html>
  );
}