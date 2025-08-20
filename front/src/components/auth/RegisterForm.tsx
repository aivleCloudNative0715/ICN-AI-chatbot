// src/components/auth/RegisterForm.tsx
'use client';

import React, { useState } from 'react';
import { InputText } from 'primereact/inputtext';
import { Password } from 'primereact/password';
import { Button } from 'primereact/button';
import { UserIcon, LockClosedIcon } from '@heroicons/react/24/outline';
import { API_BASE_URL } from '@/lib/api';
import { Checkbox } from 'primereact/checkbox'; // Checkbox 임포트
import Link from 'next/link'; // Link 임포트

interface RegisterFormProps {
  onSubmit: (data: any) => void;
  anonymousSessionId: string | null;
}

export default function RegisterForm({ onSubmit, anonymousSessionId }: RegisterFormProps) {
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [agreedToPrivacy, setAgreedToPrivacy] = useState(false)
  const [errors, setErrors] = useState<{
    userId?: string;
    password?: string;
    confirmPassword?: string;
    general?: string;
    privacy?: string; 
  }>({});

  const validate = () => {
    const newErrors: { userId?: string; password?: string; confirmPassword?: string; privacy?: string } = {};

    if (!userId) newErrors.userId = '아이디를 입력해주세요.';
    else if (!/^[a-zA-Z0-9]{6,12}$/.test(userId))
      newErrors.userId = '영문(대소문자), 숫자 조합으로 6~12자';

    if (!password) newErrors.password = '비밀번호를 입력해주세요.';
    else if (!/^(?=.*[a-zA-Z])(?=.*\d)(?=.*[!@#$%^&*()_+])[a-zA-Z\d!@#$%^&*()_+]{10,20}$/.test(password))
      newErrors.password = '영문(대소문자), 숫자, 특수문자 조합으로 10~20자 사용 가능합니다.';

    if (!confirmPassword) newErrors.confirmPassword = '비밀번호를 재입력해주세요.';
    else if (password !== confirmPassword)
      newErrors.confirmPassword = '비밀번호가 일치하지 않습니다.';

    if (!agreedToPrivacy) {
      newErrors.privacy = '개인정보 수집 및 이용에 동의해야 합니다.';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    if (validate()) {
      setErrors({});
      onSubmit({
        userId,
        password,
        passwordConfirm: confirmPassword,
        anonymousSessionId
      });
    }
  };

  const handleIdCheck = async () => {
    // 아이디 중복 확인 로직 (기존과 동일)
    if (!userId) {
      setErrors((prev) => ({ ...prev, userId: '아이디를 입력해주세요.' }));
      return;
    }
    if (!/^[a-zA-Z0-9]{6,12}$/.test(userId)) {
      setErrors((prev) => ({ ...prev, userId: '영문(대소문자), 숫자 조합으로 6~12자' }));
      return;
    }
    setErrors((prev) => ({ ...prev, userId: undefined }));

    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/check-id`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId }),
      });
      const data = await response.json();

      if (!response.ok) {
        setErrors((prev) => ({ ...prev, userId: data.message || '아이디 확인 중 오류 발생' }));
        return;
      }

      if (data.isAvailable) {
        alert('사용 가능한 아이디입니다.');
      } else {
        setErrors((prev) => ({ ...prev, userId: '이미 사용중인 아이디입니다.' }));
      }
    } catch (error) {
      console.error('아이디 중복 확인 API 호출 오류:', error);
      alert('네트워크 오류가 발생했습니다.');
    }
  };

  // Google 로그인 URL 동적 생성
    const googleLoginUrl = anonymousSessionId
    ? `${process.env.NEXT_PUBLIC_API_BASE_URL}/api/auth/oauth2/start?anonymousSessionId=${anonymousSessionId}`
    : `${process.env.NEXT_PUBLIC_API_BASE_URL}/api/auth/oauth2/start`;

  return (
    <form onSubmit={handleRegister} className="p-fluid">
      {errors.general && (
        <div className="flex items-center mb-4 p-2 text-red-500 bg-red-100 border border-red-300 rounded-md text-sm">
           <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9 13a1 1 0 112 0 1 1 0 01-2 0zm0-5a1 1 0 011-1h.01a1 1 0 110 2H10a1 1 0 01-1-1z" clipRule="evenodd" />
          </svg>
          <small>{errors.general}</small>
        </div>
      )}
      <div className="field">
        <label htmlFor="reg-userId" className="sr-only">아이디</label>
        <div className="p-inputgroup flex gap-4 items-center border-b-2 border-gray-300 focus-within:border-blue-500 transition-all duration-300">
          <span className="p-inputgroup-addon bg-white px-0 py-2">
            <UserIcon className="h-5 w-5 text-gray-400" />
          </span>
          <InputText
            id="reg-userId"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="아이디를 입력해주세요"
            className={errors.userId ? 'p-invalid' : ''}
            pt={{ root: { className: `!w-full !py-2.5 !text-base focus:shadow-none !border-none ${errors.userId ? 'p-invalid' : ''}` }}}
          />
          <Button
            label="확인"
            type="button"
            onClick={handleIdCheck}
            pt={{ root: { className: '!bg-white !text-blue-500 !border !border-blue-500 !rounded-md !px-4 !py-2 !h-auto !text-sm !font-semibold !shadow-none !whitespace-nowrap !min-w-fit' }, label: { className: 'whitespace-nowrap' }}}
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
            pt={{ root: { className: 'flex-grow' }, input: { className: `!w-full !py-2.5 !text-base focus:shadow-none !border-none ${errors.password ? 'p-invalid' : ''}` }, showIcon: { className: 'ml-auto mr-1 p-0' }, hideIcon: { className: 'ml-auto mr-1 p-0' }}}
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
            pt={{ root: { className: 'flex-grow' }, input: { className: `!w-full !py-2.5 !text-base focus:shadow-none !border-none ${errors.confirmPassword ? 'p-invalid' : ''}` }, showIcon: { className: 'ml-auto mr-1 p-0' }, hideIcon: { className: 'ml-auto mr-1 p-0' }}}
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

      {/* 개인정보 수집 및 이용 동의 체크박스 추가 */}
      <div className="field-checkbox mt-4">
        <Checkbox
          inputId="privacy-agree"
          className='border border-gray-300'
          checked={agreedToPrivacy}
          onChange={(e) => setAgreedToPrivacy(e.checked || false)}
        />
        <label htmlFor="privacy-agree" className="ml-2 text-sm">
          [필수] 
          <Link href="/privacy" target="_blank" className="text-blue-600 hover:underline">
            개인정보 수집 및 이용
          </Link>
          에 동의합니다.
        </label>
        {errors.privacy && (
          <small className="p-error block mt-1">{errors.privacy}</small>
        )}
      </div>

      <Button
        type="submit"
        label="회원가입"
        pt={{ root: { className: 'w-full mt-6 py-3 bg-[#FFC107] text-white rounded-md text-lg font-semibold shadow-md hover:bg-[#e6b000] transition-colors duration-200 !shadow-none !border-none' }}}
      />

      {/* "Google로 계속하기" 버튼 스타일 적용 */}
      {/* a 태그로 감싸서 백엔드 OAuth2 로그인 URL로 이동시킵니다. */}
      <a href={googleLoginUrl} style={{ textDecoration: 'none' }}>
        <Button
          type="button" // form submit을 방지하기 위해 type="button"으로 설정
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
      </a>
    </form>
  );
}
