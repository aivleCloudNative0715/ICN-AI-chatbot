// src/components/auth/LoginForm.tsx
'use client';

import React, { useState } from 'react';
import { InputText } from 'primereact/inputtext';
import { Password } from 'primereact/password';
import { Button } from 'primereact/button';
import { UserIcon, LockClosedIcon } from '@heroicons/react/24/outline';

interface LoginFormProps {
  onLoginSuccess: () => void;
}

export default function LoginForm({ onLoginSuccess }: LoginFormProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errors, setErrors] = useState<{ email?: string; password?: string }>({});

  const validate = () => {
    const newErrors: { email?: string; password?: string } = {};
    if (!email) newErrors.email = '아이디를 입력해주세요.';
    // 영문(대소문자), 숫자 조합으로 6~12자
    else if (!/^[a-zA-Z0-9]{6,12}$/.test(email))
      newErrors.email = '영문(대소문자), 숫자 조합으로 6~12자';

    if (!password) newErrors.password = '비밀번호를 입력해주세요.';
    // 영문 대소문자, 숫자, 특수문자 조합으로 10~20자
    else if (!/^(?=.*[a-zA-Z])(?=.*\d)(?=.*[!@#$%^&*()_+])[a-zA-Z\d!@#$%^&*()_+]{10,20}$/.test(password))
      newErrors.password = '영문(대소문자), 숫자, 특수문자 조합으로 10~20자 사용 가능합니다.';

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validate()) {
      // TODO: API 호출 로직 (API-07-19029) 구현
      console.log('로그인 시도:', { email, password });
      // 임시 성공 처리
      onLoginSuccess();
    }
  };

  return (
    <form onSubmit={handleSubmit} className="p-fluid">
      <div className="field">
        <label htmlFor="email" className="sr-only">아이디/이메일</label>
        <div className="p-inputgroup">
          <span className="p-inputgroup-addon">
            <UserIcon className="h-5 w-5 text-gray-500" />
          </span>
          <InputText
            id="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="아이디를 입력해주세요"
            className={errors.email ? 'p-invalid' : ''}
          />
        </div>
        {errors.email && <small className="p-error block mt-1">{errors.email}</small>}
      </div>

      <div className="field mt-4">
        <label htmlFor="password" className="sr-only">비밀번호</label>
        <div className="p-inputgroup">
          <span className="p-inputgroup-addon">
            <LockClosedIcon className="h-5 w-5 text-gray-500" />
          </span>
          <Password
            id="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="비밀번호를 입력해주세요"
            feedback={false}
            toggleMask
            className={errors.password ? 'p-invalid' : ''}
          />
        </div>
        {errors.password && <small className="p-error block mt-1">{errors.password}</small>}
      </div>

      <Button
        type="submit"
        label="로그인"
        className="p-button-warning w-full mt-6 py-3"
      />

      <Button
        label={
          <div className="flex items-center justify-center">
            <img src="/google-logo.svg" alt="Google" className="h-5 w-5 mr-2" />
            <span>Google로 계속하기</span>
          </div>
        }
        className="p-button-outlined p-button-secondary w-full mt-3 py-3"
        // TODO: Google 로그인 로직 (API-07-19029)
      />
    </form>
  );
}