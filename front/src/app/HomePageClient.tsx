'use client';

import ChatBotScreen, { WebSocketResponseDto } from '@/components/chat/ChatBotScreen';
import ChatSidebar from '@/components/chat/ChatSidebar';
import Header from '@/components/common/Header';
import AuthModal from '@/components/auth/AuthModal';
import { useState, useEffect, useCallback } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { API_BASE_URL } from '@/lib/api';
import { useAuth } from '@/contexts/AuthContext';

// // User와 Admin 타입을 정의합니다.
// interface LoginResponseData {
//   accessToken: string;
//   id: number;
//   userId?: string; // 일반 유저
//   googleId?: string | null; // 일반 유저
//   loginProvider?: 'LOCAL' | 'GOOGLE'; // 일반 유저
//   adminId?: string; // 관리자
//   adminName?: string; // 관리자 이름
//   role?: 'ADMIN' | 'SUPER' | 'USER'; // 관리자 또는 유저
//   sessionId: string; // 모든 로그인 응답에 세션 ID 포함
// }

// function isAdminData(data: LoginResponseData): boolean {
//   return data.role === 'ADMIN' || data.role === 'SUPER';
// }

export default function HomePageClient() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);

  const { 
    isLoggedIn, 
    isAdmin,
    login, 
    logout,
    sessionId,
    setSessionId
  } = useAuth();

  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  // 비로그인 사용자를 위한 익명 세션 ID 상태
  const [anonymousSessionId, setAnonymousSessionId] = useState<string | null>(null);
  // 로그인/비로그인 상태를 포괄하는 현재 세션 ID 상태
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  // 채팅 내역을 저장할 상태 변수
  const [chatHistory, setChatHistory] = useState<WebSocketResponseDto[]>([]);
  // OAuth 처리 중 다른 로직 실행을 막기 위한 상태 추가
  const [isProcessingOAuth, setIsProcessingOAuth] = useState(true);

  const fetchChatHistory = useCallback(async (sessionId: string) => {
    const token = localStorage.getItem('jwt_token');
    // 토큰이나 세션 ID가 없으면 함수를 즉시 종료합니다.
    if (!token || !sessionId) return;

    try {
      console.log(`[API] 채팅 내역을 요청합니다. 세션 ID: ${sessionId}`);
      const response = await fetch(`${API_BASE_URL}/api/chat/history?session_id=${sessionId}`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`, // 인증 헤더를 추가합니다.
        },
      });

      if (response.ok) {
        const historyData = await response.json();
        console.log("[API] 채팅 내역을 성공적으로 가져왔습니다:", historyData);
        setChatHistory(historyData); // 상태를 업데이트합니다.
      } else {
        console.error('채팅 내역 조회에 실패했습니다.');
        setChatHistory([]); // 실패 시 빈 배열로 초기화합니다.
      }
    } catch (error) {
      console.error('채팅 내역 요청 중 네트워크 오류가 발생했습니다:', error);
      setChatHistory([]);
    }
  }, []);

  /**
   * 로그인/회원가입 성공 시 공통 처리 핸들러
   * 응답에 포함된 accessToken과 sessionId를 Local Storage에 저장합니다.
   */
  const handleLoginSuccess = useCallback((data: any) => {
    console.log("로그인 성공 핸들러가 Context의 login 함수를 호출합니다.");
    login(data); // Context의 login 함수 호출
    setIsAuthModalOpen(false); // 모달 닫기
  }, [login]);

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
        setAnonymousSessionId(data.sessionId);
        // 비로그인 상태에서도 채팅을 이어가기 위해 세션 ID를 저장
        localStorage.setItem('session_id', data.sessionId);
        setCurrentSessionId(data.sessionId); // 상태 업데이트
      } else {
        console.error('익명 세션 ID 발급 실패');
      }
    } catch (error) {
      console.error('익명 세션 ID 요청 중 네트워크 오류:', error);
    }
  }, []);


  // useEffect(() => {
  //   const localToken = localStorage.getItem('jwt_token');
  //   const localSessionId = localStorage.getItem('session_id');

  //   // 1. OAuth2 로그인 실패 시 전달된 에러 메시지를 먼저 확인합니다.
  //   const oauthError = searchParams.get('error');
  //   if (oauthError) {
  //     alert(oauthError); // 백엔드에서 보낸 에러 메시지를 그대로 alert 창에 띄웁니다.
  //     router.replace('/'); // URL에서 에러 파라미터를 제거합니다.
  //     return; // 에러가 있으면 더 이상 진행하지 않습니다.
  //   }

  //   // 2. OAuth2 로그인 성공 시 전달된 토큰을 확인합니다.
  //   const oauthToken = searchParams.get('token');
  //   const oauthSessionId = searchParams.get('sessionId');

  //   if (oauthToken && oauthSessionId) {
  //     console.log("URL에서 OAuth 토큰과 세션 ID를 발견했습니다:", oauthToken, oauthSessionId);
  //     const fetchUserInfo = async (token: string, sessionId: string) => {
  //       try {
  //         console.log("토큰으로 사용자 정보 조회를 시작합니다...");
  //         const response = await fetch(`${API_BASE_URL}/api/users/me`, {
  //           headers: {
  //             'Authorization': `Bearer ${token}`,
  //           },
  //         });

  //         if (response.ok) {
  //           const userData = await response.json();
  //           console.log("사용자 정보를 성공적으로 가져왔습니다:", userData);
  //           handleLoginSuccess({ ...userData, accessToken: token, sessionId: sessionId });
  //           console.log("로그인 처리가 완료되어 URL을 정리합니다.");
  //           router.replace('/');
  //         } else {
  //           const errorData = await response.json();
  //           console.error('사용자 정보 조회 실패:', errorData);
  //           alert(`로그인 처리 중 오류가 발생했습니다: ${errorData.message}`);
  //           router.replace('/');
  //         }
  //       } catch (error) {
  //         console.error('사용자 정보 조회 중 네트워크 오류 발생:', error);
  //         alert('로그인 처리 중 네트워크 오류가 발생했습니다.');
  //         router.replace('/');
  //       }
  //     };

  //     fetchUserInfo(oauthToken, oauthSessionId);
      
  //   } else {
  //     // 로컬 토큰 확인 및 세션 관리 로직
  //     const localToken = localStorage.getItem('jwt_token');
  //     const localSessionId = localStorage.getItem('session_id'); // 로컬 세션 ID도 확인

  //     if (localToken && localSessionId) {
  //       // 로그인 상태일 때
  //       setIsLoggedIn(true);
  //       setCurrentSessionId(localSessionId); // 로그인 상태일 때 세션 ID 설정
  //       const userRole = localStorage.getItem('user_role');
  //       if (userRole === 'ADMIN' || userRole === 'SUPER') {
  //         setIsAdmin(true);
  //       }
  //     } else {
  //       // 비로그인 상태일 때, 익명 세션을 발급받는다.
  //       fetchAnonymousSession();
  //     }
  //   }

  //   // 일반적인 페이지 로드/새로고침 시
  //   if (localToken && localSessionId) {
  //     // 로그인 상태일 때
  //     console.log("로그인 상태를 확인했습니다. 채팅 내역 조회를 시작합니다.");
  //     setIsLoggedIn(true);
  //     setCurrentSessionId(localSessionId);
  //     const userRole = localStorage.getItem('user_role');
  //     if (userRole === 'ADMIN' || userRole === 'SUPER') setIsAdmin(true);
      
  //     // 로그인 상태이므로, 채팅 내역을 바로 조회합니다.
  //     fetchChatHistory(localSessionId);

  //   } else {
  //     // 비로그인 상태일 때
  //     console.log("비로그인 상태입니다. 익명 세션을 발급받습니다.");
  //     fetchAnonymousSession();
  //   }

  // }, [searchParams, handleLoginSuccess, router, fetchAnonymousSession, fetchChatHistory]);

    // ✅ 2. useEffect 로직을 명확하게 분리합니다.
  useEffect(() => {
    const oauthToken = searchParams.get('token');
    const oauthSessionId = searchParams.get('sessionId');
    const oauthError = searchParams.get('error');

    // --- 최우선 처리: OAuth2 리디렉션 처리 ---
    if (oauthToken && oauthSessionId) {
      console.log("URL에서 OAuth 토큰과 세션 ID를 발견. 처리 시작.");
      
      const fetchUserInfo = async (token: string, sessionId: string) => {
        try {
          const response = await fetch(`${API_BASE_URL}/api/users/me`, {
            headers: { 'Authorization': `Bearer ${token}` },
          });
          if (response.ok) {
            const userData = await response.json();
            handleLoginSuccess({ ...userData, accessToken: token, sessionId: sessionId });
          } else {
            alert('로그인 처리 중 오류가 발생했습니다.');
          }
        } catch (error) {
          console.error('사용자 정보 조회 중 네트워크 오류:', error);
        } finally {
          // URL 정리 및 처리 완료 상태로 변경
          router.replace('/');
          setIsProcessingOAuth(false);
        }
      };
      fetchUserInfo(oauthToken, oauthSessionId);
      return; // OAuth 처리 중에는 아래 로직을 실행하지 않음
    }
    
    if (oauthError) {
        alert(oauthError);
        router.replace('/');
        setIsProcessingOAuth(false);
        return;
    }

    // OAuth 파라미터가 없으면, 처리 완료 상태로 변경
    setIsProcessingOAuth(false);

  }, [searchParams, handleLoginSuccess, router]);

  const openAuthModal = (mode: 'login' | 'register' = 'login') => {
    setAuthMode(mode);
    setIsAuthModalOpen(true);
  };
  
  const closeAuthModal = () => {
    setIsAuthModalOpen(false);
  };

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  /**
   * 로그아웃 핸들러
   * 서버에 토큰 무효화를 요청하고, 클라이언트의 모든 세션 정보를 정리합니다.
   */
  const handleLogout = () => {
    logout(); // Context의 logout 함수 호출
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
      const oldSessionId = localStorage.getItem('session_id');
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

          localStorage.setItem('session_id', newSessionId);
          setCurrentSessionId(newSessionId);
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
      console.log("비회원 대화 초기화 로직을 실행합니다.");
      alert('새로운 대화를 시작합니다.');
      
      // 1. 화면의 채팅 내역을 즉시 비웁니다.
      setChatHistory([]);
      
      // 2. 새로운 익명 세션을 발급받아 세션을 교체합니다.
      await fetchAnonymousSession(); 
      
      // 3. 사이드바를 닫습니다.
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
          sessionId={currentSessionId}
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
          onLoginSuccess={handleLoginSuccess}
          initialMode={authMode}
          // 로그인 폼에 익명 세션 ID를 전달
          anonymousSessionId={anonymousSessionId}
        />
      )}
    </div>
  );
}
