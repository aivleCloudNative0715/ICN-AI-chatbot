// src/contexts/AuthContext.tsx
'use client';

import React, { createContext, useState, useContext, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { API_BASE_URL } from '@/lib/api';

// 유저 정보 타입 정의
interface User {
  id: number;
  userId: string; // 일반 유저 ID
  role: 'USER' | 'ADMIN' | 'SUPER';
  loginProvider: 'LOCAL' | 'GOOGLE';
  adminId?: string; // 관리자 ID
  adminName?: string; // 관리자 이름
}

// Context가 제공할 값들의 타입 정의
interface AuthContextType {
  isLoggedIn: boolean;
  user: User | null;
  isAdmin: boolean;
  token: string | null;
  sessionId: string | null;
  login: (loginData: any, anonymousSessionId: string | null) => Promise<boolean>;
  register: (registerData: any, anonymousSessionId: string | null) => Promise<boolean>;
  setLoginState: (loginData: any) => void;
  logout: () => void;
  initializeSession: (sessionId: string | null) => void;
 }

const AuthContext = createContext<AuthContextType | null>(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth는 반드시 AuthProvider 안에서 사용해야 합니다.');
  }
  return context;
};

// 3. Context를 제공하는 Provider 컴포넌트
export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const router = useRouter();

  // 앱이 처음 로드될 때 localStorage에서 로그인 정보를 읽어와 상태를 복원
  useEffect(() => {
    const storedToken = localStorage.getItem('jwt_token');
    const storedSessionId = localStorage.getItem('session_id');
    const storedUser = {
      id: Number(localStorage.getItem('user_id')),
      userId: localStorage.getItem('user_login_id')!,
      role: (localStorage.getItem('user_role') as User['role'])!,
      loginProvider: (localStorage.getItem('login_provider') as User['loginProvider'])!,
      adminId: localStorage.getItem('admin_id') || undefined,
      adminName: localStorage.getItem('admin_name') || undefined,
    };

    if (storedToken && storedUser.id) {
      setToken(storedToken);
      setUser(storedUser);
      setIsLoggedIn(true);
      setSessionId(storedSessionId);
    }
  }, []);

// 세션 초기화 함수 구현
  const initializeSession = useCallback((sid: string | null) => {
    if (sid) {
      console.log(`세션을 초기화합니다: ${sid}`);
      localStorage.setItem('session_id', sid);
      setSessionId(sid);
    }
  }, []);

  // 로그인 성공 후 공통 로직을 처리하는 내부 헬퍼 함수 생성
  const _handleLoginSuccess = useCallback((data: any) => {
    // 백엔드 응답 데이터를 기반으로 localStorage와 상태를 설정
    localStorage.setItem('jwt_token', data.accessToken);
    localStorage.setItem('user_id', String(data.id));
    localStorage.setItem('session_id', data.sessionId);

    const userData: User = {
        id: data.id,
        userId: data.userId,
        role: data.role || 'USER',
        loginProvider: data.loginProvider,
        adminId: data.adminId,
        adminName: data.adminName,
    };

    if (userData.role === 'ADMIN' || userData.role === 'SUPER') {
        localStorage.setItem('user_role', userData.role);
        localStorage.setItem('admin_id', userData.adminId!);
        localStorage.setItem('admin_name', userData.adminName!);
        
        // 상태 업데이트
        setToken(data.accessToken);
        setUser(userData);
        setIsLoggedIn(true);
        setSessionId(data.sessionId);

        router.push('/admin');
        return;
    } else {
        localStorage.setItem('user_role', 'USER');
        localStorage.setItem('user_login_id', userData.userId);
        localStorage.setItem('login_provider', userData.loginProvider);
    }

    setToken(data.accessToken);
    setUser(userData);
    setIsLoggedIn(true);
    setSessionId(data.sessionId);
  }, [router]);

  // OAuth 같이 이미 인증된 경우를 위한 setLoginState 함수 구현
  const setLoginState = useCallback((loginData: any) => {
    console.log("OAuth 로그인 성공. 상태를 설정합니다:", loginData);
    _handleLoginSuccess(loginData);
  }, [_handleLoginSuccess]);


// 일반 로그인을 위한 login 함수 (기존 로직 유지)
  const login = useCallback(async (loginData: any, anonymousSessionId: string | null): Promise<boolean> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...loginData,
          anonymous_session_id: anonymousSessionId,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || '로그인에 실패했습니다.');
      }
      
      const data = await response.json();
      // 성공 시 공통 로직 처리
      _handleLoginSuccess(data);
      return true;
    } catch (error) {
      alert(error instanceof Error ? error.message : '로그인 중 오류가 발생했습니다.');
      return false;
    }
  }, [_handleLoginSuccess]);

  // 회원가입 API를 호출하는 register 함수 추가
  const register = useCallback(async (registerData: any, anonymousSessionId: string | null): Promise<boolean> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/users/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...registerData,
          anonymous_session_id: anonymousSessionId, // 백엔드가 지원한다면 세션 마이그레이션을 위해 전달
        }),
      });

      if (!response.ok) { // 201 Created 외의 상태도 에러로 처리
        const errorData = await response.json();
        throw new Error(errorData.message || '회원가입에 실패했습니다.');
      }

      const data = await response.json();
      alert('회원가입 성공! 자동 로그인됩니다.');
      _handleLoginSuccess(data); // 회원가입 응답에 토큰이 포함되어 있으므로 바로 로그인 상태로 만듦
      return true; // 성공 시 true 반환
    } catch (error) {
      alert(error instanceof Error ? error.message : '회원가입 중 오류가 발생했습니다.');
      return false; // 실패 시 false 반환
    }
  }, [_handleLoginSuccess]);

  const logout = useCallback(async () => {
    const currentToken = localStorage.getItem('jwt_token');
    if (currentToken) {
        try {
            await fetch(`${API_BASE_URL}/api/users/logout`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${currentToken}` },
            });
        } catch (error) {
            console.error('서버 로그아웃 실패:', error);
        }
    }
    
    // 로컬 정보 정리
    localStorage.clear();
    setToken(null);
    setUser(null);
    setIsLoggedIn(false);
    setSessionId(null);
    alert('로그아웃되었습니다.');
    // router.push('/');
    // router.refresh();

  }, [router]);

  const value = {
    isLoggedIn,
    user,
    isAdmin: user?.role === 'ADMIN' || user?.role === 'SUPER',
    token,
    sessionId,
    initializeSession,
    login,
    setLoginState,
    logout,
    register,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};