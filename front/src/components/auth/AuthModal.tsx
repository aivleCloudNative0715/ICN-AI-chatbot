// src/components/auth/AuthModal.tsx
'use client';

import React, { useState, useEffect } from 'react';
import Modal from '@/components/common/Modal';
import LoginForm from './LoginForm';
import RegisterForm from './RegisterForm';
import { Button } from 'primereact/button';

// 로그인/회원가입 응답 타입을 하나로 통일
interface LoginResponseData {
  accessToken: string;
  id: number;
  userId?: string;
  googleId?: string | null;
  loginProvider?: 'LOCAL' | 'GOOGLE';
  adminId?: string;
  adminName?: string;
  role?: 'ADMIN' | 'SUPER' | 'USER';
  sessionId: string;
}

interface AuthModalProps {
  onClose: () => void;
  initialMode?: 'login' | 'register';
  onLoginSuccess: (data: LoginResponseData) => void;
  // 비로그인 시 사용하던 익명 세션 ID
  anonymousSessionId: string | null;
}

export default function AuthModal({ onClose, initialMode = 'login', onLoginSuccess, anonymousSessionId }: AuthModalProps) {
  const [isRegisterMode, setIsRegisterMode] = useState(initialMode === 'register');

  useEffect(() => {
    setIsRegisterMode(initialMode === 'register');
  }, [initialMode]);

  return (
    <Modal isOpen={true} onClose={onClose} title={isRegisterMode ? '회원가입' : '로그인'}>
      {isRegisterMode ? (
        // 회원가입 성공 시 onLoginSuccess를 직접 호출
        <RegisterForm onRegisterSuccess={onLoginSuccess} anonymousSessionId={anonymousSessionId}/>
      ) : (
        // 로그인 폼에 익명 세션 ID 전달
        <LoginForm onLoginSuccess={onLoginSuccess} anonymousSessionId={anonymousSessionId} />
      )}

      {isRegisterMode && (
        <div className="mt-6 pt-4 border-t border-gray-200 flex justify-end">
          <Button
            className="p-button-link text-gray-500 text-xs"
            onClick={() => setIsRegisterMode(false)}
            pt={{
              root: { className: '!p-0' },
              label: { className: 'text-gray-500 hover:text-blue-600' }
            }}
          >
            <strong>로그인</strong>으로 돌아가기
          </Button>
        </div>
      )}

      {!isRegisterMode && (
        <div className="mt-4 pt-4 border-t border-gray-200 flex justify-between items-center">
          <span className="text-gray-600 text-xs">회원가입 후 더 편리하게 이용하실 수 있습니다.</span>
          <Button
            label="회원가입"
            onClick={() => setIsRegisterMode(true)}
            pt={{
              root: { className: 'bg-gray-700 text-white rounded-full px-4 py-2 text-sm !shadow-none hover:bg-gray-800 transition-colors duration-200' }
            }}
          />
        </div>
      )}
    </Modal>
  );
}
