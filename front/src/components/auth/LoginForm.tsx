// src/components/auth/LoginForm.tsx
'use client';

import React, { useState } from 'react';
import { InputText } from 'primereact/inputtext';
import { Password } from 'primereact/password';
import { Button } from 'primereact/button';
import { UserIcon, LockClosedIcon } from '@heroicons/react/24/outline';
import { API_BASE_URL } from '@/lib/api';

interface UserLoginData {
  accessToken: string;
  id: number;
  userId: string;
  googleId: string | null;
  loginProvider: 'LOCAL' | 'GOOGLE';
}

interface AdminLoginData {
  accessToken: string;
  id: number;
  adminId: string;
  adminName: string;
  role: 'ADMIN' | 'SUPER';
}

type LoginData = UserLoginData | AdminLoginData;


interface LoginFormProps {
  onLoginSuccess: (data: LoginData) => void;
}

export default function LoginForm({ onLoginSuccess }: LoginFormProps) {
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const [errors, setErrors] = useState<{ userId?: string; password?: string; general?: string }>({});

  const validate = () => {
    const newErrors: { userId?: string; password?: string } = {};
    if (!userId) newErrors.userId = '아이디를 입력해주세요.';
    // 영문(대소문자), 숫자 조합으로 6~12자
    else if (!/^[a-zA-Z0-9]{6,12}$/.test(userId))
      newErrors.userId = '영문(대소문자), 숫자 조합으로 6~12자';

    if (!password) newErrors.password = '비밀번호를 입력해주세요.';
    // 영문 대소문자, 숫자, 특수문자 조합으로 10~20자
    else if (!/^(?=.*[a-zA-Z])(?=.*\d)(?=.*[!@#$%^&*()_+])[a-zA-Z\d!@#$%^&*()_+]{10,20}$/.test(password))
      newErrors.password = '영문(대소문자), 숫자, 특수문자 조합으로 10~20자 사용 가능합니다.';

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (validate()) {
      setErrors({}); // 이전 에러 초기화
      try {
        // API 엔드포인트는 백엔드에 맞게 조정해야 합니다.
        // DTO 이름이 LoginResponseDto인 것으로 보아 /login 엔드포인트일 가능성이 높습니다.
        const response = await fetch(`${API_BASE_URL}/auth/login`, { // 로그인 API로 수정
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ userId, password }),
        });

        const data = await response.json();

        if (response.ok) {
          onLoginSuccess(data); // 성공 시 받은 데이터 전체를 전달
        } else {
          setErrors(prev => ({ ...prev, general: data.message || '로그인에 실패했습니다. 아이디 또는 비밀번호를 확인해주세요.' }));
        }
      } catch (error) {
        console.error('로그인 중 오류 발생:', error);
        setErrors(prev => ({ ...prev, general: '네트워크 오류가 발생했습니다.' }));
      }
    }
  };

  return (
    <form onSubmit={handleSubmit} className="p-fluid">
      <div className="field">
        <label htmlFor="userId" className="sr-only">아이디/이메일</label>
        {/* 회원가입 폼과 동일한 밑줄 디자인 적용 */}
        <div className="p-inputgroup flex gap-4 items-center border-b-2 border-gray-300 focus-within:border-blue-500 transition-all duration-300">
          {/* 아이콘 애드온 테두리 제거 및 패딩 조정 */}
          <span className="p-inputgroup-addon bg-white px-0 py-2">
            <UserIcon className="h-5 w-5 text-gray-400" />
          </span>
          <InputText
            id="userId"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="아이디를 입력해주세요"
            className={errors.userId ? 'p-invalid' : ''}
            pt={{
                root: {
                    // !w-full은 InputText가 그룹 내에서 전체 너비를 차지하도록 합니다.
                    // !py-2.5 !text-base는 입력 필드의 높이와 폰트 크기를 조절합니다.
                    // focus:shadow-none !border-none은 PrimeReact 기본 스타일을 오버라이드합니다.
                    className: `!w-full !py-2.5 !text-base focus:shadow-none !border-none ${errors.userId ? 'p-invalid' : ''}`
                }
            }}
          />
        </div>
        {errors.userId && (
          <div className="flex items-center mt-1 text-red-500 text-sm">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <small className="p-error block">{errors.userId}</small>
          </div>
        )}
      </div>

      <div className="field mt-4">
        <label htmlFor="password" className="sr-only">비밀번호</label>
        {/* 회원가입 폼과 동일한 밑줄 디자인 적용 */}
        <div className="p-inputgroup flex gap-4 items-center border-b-2 border-gray-300 focus-within:border-blue-500 transition-all duration-300">
          {/* 아이콘 애드온 테두리 제거 및 패딩 조정 */}
          <span className="p-inputgroup-addon bg-white px-0 py-2">
            <LockClosedIcon className="h-5 w-5 text-gray-400" />
          </span>
          <Password
            id="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="비밀번호를 입력해주세요"
            feedback={false}
            toggleMask
            className={errors.password ? 'p-invalid' : ''}
            pt={{
                root: { className: 'flex-grow' }, // Password 컴포넌트의 root에 flex-grow 적용
                input: {
                    // 실제 input 태그에 스타일 적용
                    className: `!w-full !py-2.5 !text-base focus:shadow-none !border-none ${errors.password ? 'p-invalid' : ''}`
                },
                showIcon: { className: 'ml-auto mr-1 p-0' }, // 비밀번호 보이기 아이콘 위치 및 패딩 조정
                hideIcon: { className: 'ml-auto mr-1 p-0' } // 비밀번호 숨기기 아이콘 위치 및 패딩 조정
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

      <Button
        type="submit"
        label="로그인"
        pt={{
            root: {
                // 로그인 버튼 스타일 적용: 주황색 배경, 흰색 텍스트, 그림자 없음
                className: 'w-full mt-6 py-3 bg-[#FFC107] text-white rounded-md text-lg font-semibold shadow-md hover:bg-[#e6b000] transition-colors duration-200 !shadow-none !border-none'
            }
        }}
      />

      {/* "Google로 계속하기" 버튼 스타일 적용 */}
      <Button
        pt={{
            root: {
                className: 'w-full !shadow-none !border-none' // 기본 버튼 스타일 제거
            }
        }}
      >
        {/* 내부 div에 모든 시각적 스타일을 적용 */}
        <div className="flex items-center justify-center w-full mt-3 py-3 bg-white text-gray-700 border border-gray-300 rounded-full text-base font-semibold shadow-sm hover:bg-gray-50 transition-colors duration-200">
          <img src="/google-logo.svg" alt="Google" className="h-5 w-5 mr-2" />
          <span>Google로 계속하기</span>
        </div>
      </Button>
    </form>
  );
}