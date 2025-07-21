// src/components/auth/AuthModal.tsx
'use client';

import React, { useState } from 'react';
import Modal from '@/components/common/Modal';
import LoginForm from './LoginForm';
import RegisterForm from './RegisterForm';
import { Button } from 'primereact/button';

interface AuthModalProps {
  onClose: () => void;
  // onLoginSuccess: () => void; // 이제 layout.tsx에서 직접 관리하지 않음
}

export default function AuthModal({ onClose }: AuthModalProps) {
  const [isRegisterMode, setIsRegisterMode] = useState(false);

  const handleLoginOrRegisterSuccess = () => {
    // TODO: 실제 앱에서는 로그인/회원가입 성공 후 전역 상태 업데이트 (예: Context API)
    alert(isRegisterMode ? '회원가입 성공!' : '로그인 성공!');
    onClose(); // 성공 시 모달 닫기
  };

  return (
    <Modal isOpen={true} onClose={onClose} title={isRegisterMode ? '회원가입' : '로그인'}>
      {isRegisterMode ? (
        <RegisterForm onRegisterSuccess={handleLoginOrRegisterSuccess} />
      ) : (
        <LoginForm onLoginSuccess={handleLoginOrRegisterSuccess} />
      )}
      <div className="mt-4 text-center">
        {isRegisterMode ? (
          <Button
            label="로그인으로 돌아가기"
            className="p-button-link text-blue-500"
            onClick={() => setIsRegisterMode(false)}
          />
        ) : (
          <div className="flex justify-center items-center mt-4">
            <span className="text-gray-600 mr-2">회원가입 후 더 편리하게 이용하실 수 있습니다.</span>
            <Button
              label="회원가입"
              className="p-button-warning p-button-sm"
              onClick={() => setIsRegisterMode(true)}
            />
          </div>
        )}
      </div>
    </Modal>
  );
}