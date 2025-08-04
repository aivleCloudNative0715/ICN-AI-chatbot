// src/app/page.tsx
'use client';

import ChatBotScreen from '@/components/chat/ChatBotScreen';
import ChatSidebar from '@/components/chat/ChatSidebar';
import Header from '@/components/common/Header';
import AuthModal from '@/components/auth/AuthModal';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { API_BASE_URL } from '@/lib/api';

// User와 Admin 타입을 정의합니다.
interface UserLoginData {
  accessToken: string;
  id: number;
  userId: string;
  googleId: string | null;
  loginProvider: 'LOCAL' | 'GOOGLE';
}

interface AdminLoginData {
  accessToken: string;
  id: number;
  adminId: string;
  adminName: string;
  role: 'ADMIN' | 'SUPER';
}

type LoginData = UserLoginData | AdminLoginData;

// 타입 가드 함수: LoginData가 AdminLoginData인지 확인
function isAdminData(data: LoginData): data is AdminLoginData {
  return (data as AdminLoginData).role !== undefined;
}

export default function HomePage() {
  const router = useRouter();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const [isAdmin, setIsAdmin] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem('jwt_token');
    const userRole = localStorage.getItem('user_role');

    if (token) {
      setIsLoggedIn(true);
      if (userRole === 'ADMIN' || userRole === 'SUPER') {
        setIsAdmin(true);
      }
    }
  }, []);

  const openAuthModal = (mode: 'login' | 'register' = 'login') => {
    setAuthMode(mode);
    setIsAuthModalOpen(true);
  };
  
  const closeAuthModal = () => {
    setIsAuthModalOpen(false);
  };

  // 로그인/회원가입 성공 시 최종적으로 호출되는 함수
  const handleLoginSuccess = (data: LoginData) => {
    // 공통적으로 토큰과 ID 저장
    localStorage.setItem('jwt_token', data.accessToken);
    localStorage.setItem('user_id', String(data.id));

    // 타입 가드를 사용하여 관리자인지 일반 사용자인지 확인
    if (isAdminData(data)) {
      // 관리자일 경우
      localStorage.setItem('user_role', data.role);
      localStorage.setItem('admin_id', data.adminId);
      localStorage.setItem('admin_name', data.adminName);
      setIsAdmin(true);
      // 관리자 페이지로 리디렉션
      router.push('/admin');
    } else {
      // 일반 사용자일 경우
      localStorage.setItem('user_role', 'USER'); // 기본 역할 'USER'로 저장
      localStorage.setItem('user_login_id', data.userId);
      localStorage.setItem('login_provider', data.loginProvider);
      setIsAdmin(false);
    }
    
    // 로그인 상태로 변경하고 모달을 닫습니다.
    setIsLoggedIn(true);
    setIsAuthModalOpen(false);
  };

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  const handleLogout = async () => {
    const token = localStorage.getItem('jwt_token');

    if (token) {
        try {
            // 헤더에 Bearer 토큰을 담아 로그아웃 API 호출
            const response = await fetch(`${API_BASE_URL}/users/logout`, {
                method: 'POST', // 일반적으로 로그아웃은 POST를 사용합니다.
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
            });

            if (!response.ok) {
                // 서버에서 로그아웃 실패 시 에러를 콘솔에 출력합니다.
                // 클라이언트에서는 어쨌든 로그아웃 처리를 계속 진행합니다.
                const errorData = await response.json();
                console.error('Server logout failed:', errorData.message || 'Unknown error');
            }
        } catch (error) {
            console.error('Logout API call failed:', error);
        }
    }

    // API 호출 결과와 관계없이 localStorage에서 모든 관련 정보 삭제
    localStorage.removeItem('jwt_token');
    localStorage.removeItem('user_id');
    localStorage.removeItem('user_role');
    localStorage.removeItem('admin_id');
    localStorage.removeItem('admin_name');
    localStorage.removeItem('user_login_id');
    localStorage.removeItem('login_provider');
    
    // 상태를 업데이트하고 사용자에게 알림
    setIsLoggedIn(false);
    setIsAdmin(false);
    alert('로그아웃되었습니다.');
    
    // 메인 페이지로 리디렉션
    router.push('/');
  };

  return (
    <div className="flex flex-col flex-1 w-full h-full">
      <Header
        onLoginClick={() => openAuthModal('login')}
        onRegisterClick={() => openAuthModal('register')}
        onMenuClick={toggleSidebar}
        isLoggedIn={isLoggedIn}
        onLogoutClick={handleLogout}
      />
      <div className="flex-grow flex overflow-hidden">
        <ChatBotScreen
          isLoggedIn={isLoggedIn}
          onLoginStatusChange={setIsLoggedIn}
          onSidebarToggle={toggleSidebar}
        />
      </div>
      
      {isSidebarOpen && (
        <ChatSidebar isLoggedIn={isLoggedIn} onClose={toggleSidebar} />
      )}

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
