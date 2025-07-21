// src/components/auth/RegisterForm.tsx
'use client';

import React, { useState } from 'react';
import { InputText } from 'primereact/inputtext';
import { Password } from 'primereact/password';
import { Button } from 'primereact/button';
import { UserIcon, LockClosedIcon } from '@heroicons/react/24/outline';

interface RegisterFormProps {
  onRegisterSuccess: () => void;
}

export default function RegisterForm({ onRegisterSuccess }: RegisterFormProps) {
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

  const handleRegister = (e: React.FormEvent) => {
    e.preventDefault();
    if (validate()) {
      // TODO: API 호출 로직 (API-11-24027)
      console.log('회원가입 시도:', { email, password });
      // 임시 성공 처리
      onRegisterSuccess();
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
        <div className="p-inputgroup">
          <span className="p-inputgroup-addon">
            <UserIcon className="h-5 w-5 text-gray-500" />
          </span>
          <InputText
            id="reg-email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="아이디를 입력해주세요"
            className={errors.email ? 'p-invalid' : ''}
          />
          <Button
            label="확인"
            className="p-button-secondary p-button-sm"
            onClick={handleEmailCheck}
            type="button"
          />
        </div>
        {errors.email && <small className="p-error block mt-1">{errors.email}</small>}
      </div>

      <div className="field mt-4">
        <label htmlFor="reg-password" className="sr-only">비밀번호</label>
        <div className="p-inputgroup">
          <span className="p-inputgroup-addon">
            <LockClosedIcon className="h-5 w-5 text-gray-500" />
          </span>
          <Password
            id="reg-password"
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

      <div className="field mt-4">
        <label htmlFor="reg-confirm-password" className="sr-only">비밀번호 재확인</label>
        <div className="p-inputgroup">
          <span className="p-inputgroup-addon">
            <LockClosedIcon className="h-5 w-5 text-gray-500" />
          </span>
          <Password
            id="reg-confirm-password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            placeholder="비밀번호를 재입력해주세요"
            feedback={false}
            toggleMask
            className={errors.confirmPassword ? 'p-invalid' : ''}
          />
        </div>
        {errors.confirmPassword && <small className="p-error block mt-1">{errors.confirmPassword}</small>}
      </div>

      <Button
        type="submit"
        label="회원가입"
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
        // TODO: Google 회원가입 로직 (API-07-19029)
      />
    </form>
  );
}