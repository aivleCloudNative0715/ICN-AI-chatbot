// src/components/auth/AuthModal.tsx
'use client';

import React, { useState, useEffect } from 'react';
import Modal from '@/components/common/Modal';
import LoginForm from './LoginForm';
import RegisterForm from './RegisterForm';
import { Button } from 'primereact/button';

interface AuthModalProps {
  onClose: () => void;
  initialMode?: 'login' | 'register';
  onLoginSubmit: (data: any) => void;
  onRegisterSubmit: (data: any) => void;
  anonymousSessionId: string | null;
}

export default function AuthModal({ onClose, initialMode = 'login', onLoginSubmit, onRegisterSubmit, anonymousSessionId }: AuthModalProps) {
  const [isRegisterMode, setIsRegisterMode] = useState(initialMode === 'register');

  useEffect(() => {
    setIsRegisterMode(initialMode === 'register');
  }, [initialMode]);

  return (
    <Modal isOpen={true} onClose={onClose} title={isRegisterMode ? '회원가입' : '로그인'}>
      {isRegisterMode ? (
        <RegisterForm onSubmit={onRegisterSubmit} anonymousSessionId={anonymousSessionId}/>
      ) : (
        <LoginForm onSubmit={onLoginSubmit} anonymousSessionId={anonymousSessionId} />
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
