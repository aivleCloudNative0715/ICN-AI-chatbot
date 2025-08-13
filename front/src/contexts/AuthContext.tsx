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
  setSessionId: (id: string | null) => void;
  login: (data: any) => void; // 로그인 처리 함수
  logout: () => void; // 로그아웃 처리 함수
}

// 1. Context 생성 (기본값은 null)
const AuthContext = createContext<AuthContextType | null>(null);

// 2. 다른 컴포넌트에서 Context를 쉽게 사용하기 위한 Custom Hook
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

  const login = useCallback((data: any) => {
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
    } else {
        localStorage.setItem('user_role', 'USER');
        localStorage.setItem('user_login_id', userData.userId);
        localStorage.setItem('login_provider', userData.loginProvider);
    }

    setToken(data.accessToken);
    setUser(userData);
    setIsLoggedIn(true);
    setSessionId(data.sessionId);

    if (userData.role === 'ADMIN' || userData.role === 'SUPER') {
        router.push('/admin');
    }
  }, [router]);

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
    // 페이지를 새로고침하여 초기 상태로 돌아감
    router.push('/');
    router.refresh();

  }, [router]);

  const value = {
    isLoggedIn,
    user,
    isAdmin: user?.role === 'ADMIN' || user?.role === 'SUPER',
    token,
    sessionId,
    setSessionId,
    login,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};