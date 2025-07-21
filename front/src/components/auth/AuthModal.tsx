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
}

export default function AuthModal({ onClose, initialMode = 'login' }: AuthModalProps) {
  const [isRegisterMode, setIsRegisterMode] = useState(initialMode === 'register');

  useEffect(() => {
    setIsRegisterMode(initialMode === 'register');
  }, [initialMode]);

  const handleRegisterSuccessAndSwitchToLogin = () => {
    alert('회원가입 성공! 이제 로그인해주세요.');
    setIsRegisterMode(false); // 로그인 폼으로 전환
  };

  const handleLoginSuccess = () => {
    alert('로그인 성공!');
    onClose(); // 로그인 성공 시 모달 닫기
  };

  const handleOpenLoginModalFromRegister = () => {
    setIsRegisterMode(false); // 회원가입 완료 후 로그인 폼으로 전환
  };

  return (
    <Modal isOpen={true} onClose={onClose} title={isRegisterMode ? '회원가입' : '로그인'}>
      {isRegisterMode ? (
        <RegisterForm
          onRegisterSuccess={handleRegisterSuccessAndSwitchToLogin}
          onOpenLoginModal={handleOpenLoginModalFromRegister}
        />
      ) : (
        <LoginForm onLoginSuccess={handleLoginSuccess} />
      )}

      {/* "로그인으로 돌아가기" 섹션 */}
      {isRegisterMode && (
        <div className="mt-6 pt-4 border-t border-gray-200 flex justify-end"> {/* 회색 줄과 오른쪽 정렬 */}
          <Button
            // label prop 대신 children을 사용하여 텍스트 스타일을 유연하게 제어
            // p-button-link 클래스는 PrimeReact의 기본 링크 스타일을 적용합니다.
            // Tailwind 클래스로 폰트 크기, 색상, 정렬 등 조절
            className="p-button-link text-gray-500 text-xs" // p-button-link 유지, 폰트 크기 조정
            onClick={() => setIsRegisterMode(false)}
            pt={{
              root: {
                // PrimeReact 버튼의 기본 패딩/여백을 제거하고 Tailwind로 제어
                className: '!p-0'
              },
              label: {
                  className: 'text-gray-500 hover:text-blue-600' // 전체 텍스트 색상과 호버 색상
              }
            }}
          >
            <strong>로그인</strong>으로 돌아가기
          </Button>
        </div>
      )}

      {/* "회원가입 후 더 편리하게 이용하실 수 있습니다." 섹션 */}
      {!isRegisterMode && (
        <div className="mt-4 pt-4 border-t border-gray-200 flex justify-between items-center"> {/* flex justify-between 추가 */}
          <span className="text-gray-600 text-xs">회원가입 후 더 편리하게 이용하실 수 있습니다.</span> {/* 폰트 크기 조정 */}
          <Button
            label="회원가입"
            // 기존 className 제거하고 pt prop을 사용하여 스타일 적용
            onClick={() => setIsRegisterMode(true)}
            pt={{
              root: {
                className: 'bg-gray-700 text-white rounded-full px-4 py-2 text-sm !shadow-none hover:bg-gray-800 transition-colors duration-200' // 버튼 스타일 변경
              }
            }}
          />
        </div>
      )}
    </Modal>
  );
}