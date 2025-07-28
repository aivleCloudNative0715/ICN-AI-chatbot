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
  onLoginSuccess?: () => void; // 추가: 로그인 성공 콜백
}

export default function AuthModal({ onClose, initialMode = 'login', onLoginSuccess }: AuthModalProps) {
  const [isRegisterMode, setIsRegisterMode] = useState(initialMode === 'register');

  useEffect(() => {
    setIsRegisterMode(initialMode === 'register');
  }, [initialMode]);

  const handleRegisterSuccessAndSwitchToLogin = () => {
    alert('회원가입 성공! 이제 로그인해주세요.');
    setIsRegisterMode(false); // 로그인 폼으로 전환
  };

  const handleLoginSuccessInternal = () => { // 이름을 handleLoginSuccessInternal로 변경하여 prop과 혼동 방지
    // alert('로그인 성공!'); // 기존 alert는 제거하거나 onLoginSuccess 콜백 내에서 처리
    if (onLoginSuccess) { // onLoginSuccess prop이 존재하면 호출
      onLoginSuccess();
    }
    // onClose(); // 모달 닫기 로직은 onLoginSuccess 콜백 내에서 처리될 수 있음 (layout.tsx 참고)
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
        <LoginForm onLoginSuccess={handleLoginSuccessInternal} />
      )}

      {/* "로그인으로 돌아가기" 섹션 */}
      {isRegisterMode && (
        <div className="mt-6 pt-4 border-t border-gray-200 flex justify-end"> {/* 회색 줄과 오른쪽 정렬 */}
          <Button
            className="p-button-link text-gray-500 text-xs" // p-button-link 유지, 폰트 크기 조정
            onClick={() => setIsRegisterMode(false)}
            pt={{
              root: {
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