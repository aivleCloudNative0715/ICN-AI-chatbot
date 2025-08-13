'use client';

import ChatBotScreen, { WebSocketResponseDto } from '@/components/chat/ChatBotScreen';
import ChatSidebar from '@/components/chat/ChatSidebar';
import Header from '@/components/common/Header';
import AuthModal from '@/components/auth/AuthModal';
import { useState, useEffect, useCallback, useRef } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { API_BASE_URL } from '@/lib/api';
import { useAuth } from '@/contexts/AuthContext';

export default function HomePageClient() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');

  const { 
    isLoggedIn, 
    sessionId,
    login,
    logout,
    initializeSession,
    setLoginState,
  } = useAuth();

  const [chatHistory, setChatHistory] = useState<WebSocketResponseDto[]>([]);
  const [anonymousSessionId, setAnonymousSessionId] = useState<string | null>(null);
  const prevIsLoggedInRef = useRef(isLoggedIn);

  const fetchChatHistory = useCallback(async (sid: string) => {
    const token = localStorage.getItem('jwt_token');
    
    // JWT 토큰이 없으면(비회원) API를 호출하지 않고 즉시 종료합니다.
    if (!token) {
      setChatHistory([]); // 비회원이므로 채팅 기록을 항상 비워둡니다.
      return;
    }

    // 세션 ID가 없어도 종료합니다. (이 경우는 거의 발생하지 않음)
    if (!sid) {
      setChatHistory([]);
      return;
    }

    try {
      console.log(`[API] 채팅 내역 요청. 세션 ID: ${sid}`);
      const response = await fetch(`${API_BASE_URL}/api/chat/history?session_id=${sid}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`, // 토큰이 항상 있는 경우에만 호출됨
        },
      });

      if (response.ok) {
        const historyData = await response.json();
        setChatHistory(historyData);
      } else {
        console.error('채팅 내역 조회에 실패했습니다.');
        setChatHistory([]);
      }
    } catch (error) {
      console.error('채팅 내역 요청 중 네트워크 오류:', error);
      setChatHistory([]);
    }
  }, []);

  /**
   * 비로그인 사용자를 위한 익명 세션 ID를 발급받는 함수
   */
  const fetchAnonymousSession = useCallback(async () => {
    try {
      console.log("익명 세션 ID를 요청합니다: ", `${API_BASE_URL}`);
      const response = await fetch(`${API_BASE_URL}/api/chat/session`, {
        method: 'POST',
      });
      if (response.ok) {
        const data = await response.json();
        console.log("익명 세션 ID 발급 성공:", data.sessionId);
        // Context의 세션 초기화 함수를 호출하여 상태를 중앙에서 관리
        initializeSession(data.sessionId);
        // 로그인 모달에 전달하기 위해 별도 상태에도 저장
        setAnonymousSessionId(data.sessionId);
      } else {
        console.error('익명 세션 ID 발급 실패');
      }
    } catch (error) {
      console.error('익명 세션 ID 요청 중 네트워크 오류:', error);
    }
  }, [initializeSession]);

  // 컴포넌트 마운트 시 실행되는 초기화 로직
  useEffect(() => {
    // 1. OAuth 로그인 처리 (가장 먼저)
    const oauthToken = searchParams.get('token');
    const oauthSessionId = searchParams.get('sessionId');
    if (oauthToken && oauthSessionId) {
      const fetchUserInfo = async () => {
        const response = await fetch(`${API_BASE_URL}/api/users/me`, {
          headers: { 'Authorization': `Bearer ${oauthToken}` },
        });
        if (response.ok) {
          const userData = await response.json();
          setLoginState({ ...userData, accessToken: oauthToken, sessionId: oauthSessionId });
          router.replace('/');
        }
      };
      fetchUserInfo();
      return; // OAuth 처리 중에는 아래 로직을 실행하지 않음
    }

    // 2. 일반 세션 처리
    const localToken = localStorage.getItem('jwt_token');

   if (!localToken) {
      // 비로그인 상태면 무조건 새 익명 세션을 발급받습니다.
      console.log("비로그인 상태입니다. 새 익명 세션을 발급합니다.");
      fetchAnonymousSession();
    }
  }, []);

  // 로그아웃 처리를 위한 useEffect
  // isLoggedIn 상태가 false로 변경될 때를 감지하여 화면을 정리하고 새 익명 세션을 받습니다.
  useEffect(() => {
    if (prevIsLoggedInRef.current && !isLoggedIn) {
      console.log("로그아웃 감지. 채팅 기록을 비우고 새 익명 세션을 요청합니다.");
      setChatHistory([]); // 화면의 채팅 기록 즉시 비우기
      fetchAnonymousSession(); // 새 익명 세션 발급받기
    }
    // 현재 상태를 ref에 업데이트하여 다음 렌더링 시 비교할 수 있도록 함
    prevIsLoggedInRef.current = isLoggedIn;
  }, [isLoggedIn, fetchAnonymousSession]);


  // 세션 ID가 변경될 때마다 채팅 기록을 다시 불러옵니다. (기존과 동일)
  useEffect(() => {
    if (sessionId) {
      fetchChatHistory(sessionId);
    }
  }, [sessionId, fetchChatHistory]);

  const openAuthModal = (mode: 'login' | 'register' = 'login') => {
    setAuthMode(mode);
    setIsAuthModalOpen(true);
  };
  
  const closeAuthModal = () => setIsAuthModalOpen(false);
  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);
  
  // 로그아웃 핸들러는 이제 Context의 logout 함수만 호출합니다.
  const handleLogout = () => {
    logout();
  };

  const handleDeleteAccount = async () => {
    if (!window.confirm('정말로 계정을 삭제하시겠습니까?\n삭제된 계정은 복구할 수 없습니다.')) {
      return;
    }

    const token = localStorage.getItem('jwt_token');
    if (!token) {
      alert('로그인 정보가 없습니다. 다시 로그인해주세요.');
      logout(); 
      return;
    }

    try {
      // 1. 백엔드에 계정 삭제 API를 호출합니다. (이 API가 서버의 토큰도 무효화합니다)
      const response = await fetch(`${API_BASE_URL}/api/users/me`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        alert('계정이 성공적으로 삭제되었습니다.');
        logout();
      } else {
        const errorData = await response.json();
        throw new Error(errorData.message || '계정 삭제에 실패했습니다.');
      }
    } catch (error) {
      console.error('계정 삭제 중 오류 발생:', error);
      alert(error instanceof Error ? error.message : '계정 삭제 중 오류가 발생했습니다.');
    }
  };

  /**
   * 채팅 기록 초기화(리셋) 핸들러 함수를 추가
   */
  const handleClearChatHistory = async () => {
    if (!window.confirm('대화 기록을 초기화하고 새 대화를 시작하시겠습니까?')) {
      return;
    }

    const token = localStorage.getItem('jwt_token');
    // 로그인 상태에 따라 로직을 분기
    if (token) {
      // --- 회원일 경우: reset API 호출 ---
      console.log("회원 대화 초기화 로직을 실행합니다.");
      const oldSessionId = sessionId;
      try {
        const response = await fetch(`${API_BASE_URL}/api/chat/history/reset`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ old_session_id: oldSessionId }),
        });

        if (response.ok) {
          const data = await response.json();
          const newSessionId = data.new_session_id;

          alert('대화 기록이 초기화되었습니다.');

          initializeSession(newSessionId);
          setChatHistory([]);
          setIsSidebarOpen(false);
        } else {
          const errorData = await response.json();
          throw new Error(errorData.message || '기록 초기화에 실패했습니다.');
        }
      } catch (error) {
        console.error('채팅 기록 초기화 중 오류 발생:', error);
        alert(error instanceof Error ? error.message : '기록 초기화 중 오류가 발생했습니다.');
      }
    } else {
      // --- 비회원일 경우: 새 익명 세션 발급 ---
      alert('새로운 대화를 시작합니다.');
      setChatHistory([]);
      await fetchAnonymousSession(); 
      setIsSidebarOpen(false);
    }
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
          sessionId={sessionId}
          initialHistory={chatHistory}
        />
      </div>
      
      {isSidebarOpen && (
        <ChatSidebar 
          isLoggedIn={isLoggedIn}
          onClose={toggleSidebar}
          onDeleteAccount={handleDeleteAccount}
          onClearChatHistory={handleClearChatHistory}/>
      )}

      {isAuthModalOpen && (
        <AuthModal
          onClose={closeAuthModal}
          onLoginSuccess={(data) => login(data, anonymousSessionId)}
          initialMode={authMode}
          anonymousSessionId={anonymousSessionId}
        />
      )}
    </div>
  );
}
