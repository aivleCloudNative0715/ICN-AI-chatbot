// src/components/auth/RegisterForm.tsx
'use client';

import React, { useState } from 'react';
import { InputText } from 'primereact/inputtext';
import { Password } from 'primereact/password';
import { Button } from 'primereact/button';
import { UserIcon, LockClosedIcon } from '@heroicons/react/24/outline';

interface RegisterFormProps {
  onRegisterSuccess: () => void;
  onOpenLoginModal: () => void; // 추가: 로그인 모달을 열기 위한 콜백
}

export default function RegisterForm({ onRegisterSuccess, onOpenLoginModal }: RegisterFormProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [errors, setErrors] = useState<{
    email?: string;
    password?: string;
    confirmPassword?: string;
  }>({});

  const validate = () => {
    const newErrors: { email?: string; password?: string; confirmPassword?: string } = {};

    if (!email) newErrors.email = '아이디를 입력해주세요.';
    // 영문(대소문자), 숫자 조합으로 6~12자
    else if (!/^[a-zA-Z0-9]{6,12}$/.test(email))
      newErrors.email = '영문(대소문자), 숫자 조합으로 6~12자';
    // TODO: API-11-24028 아이디 중복 확인 기능 구현 필요

    if (!password) newErrors.password = '비밀번호를 입력해주세요.';
    // 영문 대소문자, 숫자, 특수문자 조합으로 10~20자
    else if (!/^(?=.*[a-zA-Z])(?=.*\d)(?=.*[!@#$%^&*()_+])[a-zA-Z\d!@#$%^&*()_+]{10,20}$/.test(password))
      newErrors.password = '영문(대소문자), 숫자, 특수문자 조합으로 10~20자 사용 가능합니다.';

    if (!confirmPassword) newErrors.confirmPassword = '비밀번호를 재입력해주세요.';
    else if (password !== confirmPassword)
      newErrors.confirmPassword = '비밀번호가 일치하지 않습니다.';

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    if (validate()) {
      // TODO: API 호출 로직 (API-11-24027)
      console.log('회원가입 시도:', { email, password });
      // 실제 API 호출이 성공했다고 가정
      try {
        // const response = await fetch('/api/register', {
        //   method: 'POST',
        //   headers: { 'Content-Type': 'application/json' },
        //   body: JSON.stringify({ email, password }),
        // });
        // if (response.ok) {
            onRegisterSuccess(); // 회원가입 모달 닫기
            onOpenLoginModal(); // 로그인 모달 열기
        // } else {
        //   const errorData = await response.json();
        //   setErrors(prev => ({ ...prev, general: errorData.message || '회원가입 실패' }));
        // }
      } catch (error) {
        console.error('회원가입 중 오류 발생:', error);
        setErrors(prev => ({ ...prev, general: '네트워크 오류가 발생했습니다.' }));
      }
    }
  };

  const handleEmailCheck = () => {
    if (!email) {
      setErrors((prev) => ({ ...prev, email: '아이디를 입력해주세요.' }));
      return;
    }
    if (!/^[a-zA-Z0-9]{6,12}$/.test(email)) {
      setErrors((prev) => ({ ...prev, email: '영문(대소문자), 숫자 조합으로 6~12자' }));
      return;
    }
    // TODO: API-11-24028 아이디 중복 확인 API 호출
    // 임시 로직:
    const isDuplicate = email === 'testuser'; // 예시
    if (isDuplicate) {
      setErrors((prev) => ({ ...prev, email: '이미 사용중인 아이디입니다.' }));
    } else {
      alert('사용 가능한 아이디입니다.');
      setErrors((prev) => ({ ...prev, email: undefined }));
    }
  };

  return (
    <form onSubmit={handleRegister} className="p-fluid">
      <div className="field">
        <label htmlFor="reg-email" className="sr-only">아이디/이메일</label>
        <div className="p-inputgroup flex gap-4 items-center border-b-2 border-gray-300 focus-within:border-blue-500 transition-all duration-300">
          <span className="p-inputgroup-addon bg-white px-0 py-2">
            <UserIcon className="h-5 w-5 text-gray-400" />
          </span>
          <InputText
            id="reg-email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="아이디를 입력해주세요"
            className={errors.email ? 'p-invalid' : ''}
            pt={{
                root: {
                    className: `!w-full !py-2.5 !text-base focus:shadow-none !border-none ${errors.email ? 'p-invalid' : ''}`
                }
            }}
          />
          <Button
            label="확인"
            type="button"
            onClick={handleEmailCheck}
            pt={{
                root: {
                    className: '!bg-white !text-blue-500 !border !border-blue-500 !rounded-md !px-4 !py-2 !h-auto !text-sm !font-semibold !shadow-none !whitespace-nowrap !min-w-fit'
                },
                label: {
                    className: 'whitespace-nowrap'
                }
            }}
          />
        </div>
        {errors.email && (
          <div className="flex items-center mt-1 text-red-500 text-sm">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <small className="p-error block">{errors.email}</small>
          </div>
        )}
      </div>

      <div className="field mt-4">
        <label htmlFor="reg-password" className="sr-only">비밀번호</label>
        <div className="p-inputgroup flex gap-4 items-center border-b-2 border-gray-300 focus-within:border-blue-500 transition-all duration-300">
          <span className="p-inputgroup-addon bg-white px-0 py-2">
            <LockClosedIcon className="h-5 w-5 text-gray-400" />
          </span>
          <Password
            id="reg-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="비밀번호를 입력해주세요"
            feedback={false}
            toggleMask
            className={errors.password ? 'p-invalid' : ''}
            pt={{
                root: { className: 'flex-grow' },
                input: {
                    className: `!w-full !py-2.5 !text-base focus:shadow-none !border-none ${errors.password ? 'p-invalid' : ''}`
                },
                showIcon: { className: 'ml-auto mr-1 p-0' },
                hideIcon: { className: 'ml-auto mr-1 p-0' }
            }}
          />
        </div>
        {errors.password && (
          <div className="flex items-center mt-1 text-red-500 text-sm">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <small className="p-error block">{errors.password}</small>
          </div>
        )}
      </div>

      <div className="field mt-4">
        <label htmlFor="reg-confirm-password" className="sr-only">비밀번호 재확인</label>
        <div className="p-inputgroup flex gap-4 items-center border-b-2 border-gray-300 focus-within:border-blue-500 transition-all duration-300">
          <span className="p-inputgroup-addon bg-white px-0 py-2">
            <LockClosedIcon className="h-5 w-5 text-gray-400" />
          </span>
          <Password
            id="reg-confirm-password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            placeholder="비밀번호를 재입력해주세요"
            feedback={false}
            toggleMask
            className={errors.confirmPassword ? 'p-invalid' : ''}
            pt={{
                root: { className: 'flex-grow' },
                input: {
                    className: `!w-full !py-2.5 !text-base focus:shadow-none !border-none ${errors.confirmPassword ? 'p-invalid' : ''}`
                },
                showIcon: { className: 'ml-auto mr-1 p-0' },
                hideIcon: { className: 'ml-auto mr-1 p-0' }
            }}
          />
        </div>
        {errors.confirmPassword && (
          <div className="flex items-center mt-1 text-red-500 text-sm">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <small className="p-error block">{errors.confirmPassword}</small>
          </div>
        )}
      </div>

      <Button
        type="submit"
        label="회원가입"
        pt={{
            root: {
                className: 'w-full mt-6 py-3 bg-[#FFC107] text-white rounded-md text-lg font-semibold shadow-md hover:bg-[#e6b000] transition-colors duration-200 !shadow-none !border-none'
            }
        }}
      />

      <Button
        // pt.root에 테두리 및 그림자 관련 클래스만 남기고,
        // 나머지 시각적 스타일은 내부 div에 그대로 적용
        pt={{
            root: {
                className: 'w-full !shadow-none !border-none'
                // !border-width-[1px]는 Tailwind 기본이 아니므로 제거 또는 custom config 필요
            }
        }}
      >
        {/* 이 div에 버튼의 배경, 패딩, 텍스트 색상, 테두리, 둥근 모서리, 그림자 등 모든 스타일을 적용합니다. */}
        <div className="flex items-center justify-center w-full mt-3 py-3 bg-white text-gray-700 border border-gray-300 rounded-full text-base font-semibold shadow-sm hover:bg-gray-50 transition-colors duration-200">
          <img src="/google-logo.svg" alt="Google" className="h-5 w-5 mr-2" />
          <span>Google로 계속하기</span>
        </div>
      </Button>
    </form>
  );
}