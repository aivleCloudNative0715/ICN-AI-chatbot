// src/app/page.tsx
'use client';

import ChatBotScreen from '@/components/chat/ChatBotScreen';
import ChatSidebar from '@/components/chat/ChatSidebar';
import Header from '@/components/common/Header';
import AuthModal from '@/components/auth/AuthModal';
import { useState, useEffect, useCallback } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
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
  const searchParams = useSearchParams();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const [isAdmin, setIsAdmin] = useState(false);

  const handleLoginSuccess = useCallback((data: LoginData) => {
    console.log("handleLoginSuccess 호출됨. Local Storage에 정보를 저장합니다.");
    localStorage.setItem('jwt_token', data.accessToken);
    localStorage.setItem('user_id', String(data.id));

    if (isAdminData(data)) {
      localStorage.setItem('user_role', data.role);
      localStorage.setItem('admin_id', data.adminId);
      localStorage.setItem('admin_name', data.adminName);
      setIsAdmin(true);
      router.push('/admin');
    } else {
      localStorage.setItem('user_role', 'USER');
      localStorage.setItem('user_login_id', data.userId);
      localStorage.setItem('login_provider', data.loginProvider);
      setIsAdmin(false);
    }
    
    setIsLoggedIn(true);
    setIsAuthModalOpen(false);
  }, [router]);


  useEffect(() => {
    const oauthToken = searchParams.get('token');

    if (oauthToken) {
      console.log("URL에서 OAuth 토큰을 발견했습니다:", oauthToken);
      const fetchUserInfo = async (token: string) => {
        try {
          console.log("토큰으로 사용자 정보 조회를 시작합니다...");
          const response = await fetch(`${API_BASE_URL}/users/me`, {
            headers: {
              // 이 부분은 문제가 없습니다.
              'Authorization': `Bearer ${token}`,
            },
          });

          if (response.ok) {
            const userData = await response.json();
            console.log("사용자 정보를 성공적으로 가져왔습니다:", userData);
            handleLoginSuccess({ ...userData, accessToken: token });
            console.log("로그인 처리가 완료되어 URL을 정리합니다.");
            router.replace('/');
          } else {
            const errorData = await response.json();
            console.error('사용자 정보 조회 실패:', errorData);
            alert(`로그인 처리 중 오류가 발생했습니다: ${errorData.message}`);
            router.replace('/');
          }
        } catch (error) {
          console.error('사용자 정보 조회 중 네트워크 오류 발생:', error);
          alert('로그인 처리 중 네트워크 오류가 발생했습니다.');
          router.replace('/');
        }
      };

      fetchUserInfo(oauthToken);
      
    } else {
      const localToken = localStorage.getItem('jwt_token');
      if (localToken) {
        setIsLoggedIn(true);
        const userRole = localStorage.getItem('user_role');
        if (userRole === 'ADMIN' || userRole === 'SUPER') {
          setIsAdmin(true);
        }
      }
    }
  }, [searchParams, handleLoginSuccess, router]);

  const openAuthModal = (mode: 'login' | 'register' = 'login') => {
    setAuthMode(mode);
    setIsAuthModalOpen(true);
  };
  
  const closeAuthModal = () => {
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
