'use client';

import React, { useState } from 'react';
import { InputText } from 'primereact/inputtext';
import { Password } from 'primereact/password';
import { Dropdown } from 'primereact/dropdown';
import { Button } from 'primereact/button';
import { UserIcon, LockClosedIcon, XMarkIcon } from '@heroicons/react/24/outline';

interface AdminCreateModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreate: (data: { email: string; name: string; role: string }) => void;
}

export default function AdminCreateModal({ isOpen, onClose, onCreate }: AdminCreateModalProps) {
  const [email, setEmail] = useState('');
  const [isEmailChecked, setIsEmailChecked] = useState(false);
  const [emailAvailable, setEmailAvailable] = useState<boolean | null>(null);
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [name, setName] = useState('');
  const [role, setRole] = useState('관리자');
  const [errors, setErrors] = useState<{ email?: string; password?: string; confirmPassword?: string; name?: string }>({});

  const isValidEmail = (value: string) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
  const isValidPassword = (value: string) =>
    /^(?=.*[a-zA-Z])(?=.*\d)(?=.*[!@#$%^&*()_+])[a-zA-Z\d!@#$%^&*()_+]{10,20}$/.test(value);

  const handleCheckEmail = () => {
    if (!email) {
      setErrors((prev) => ({ ...prev, email: '아이디를 입력해주세요.' }));
      return;
    }
    if (!isValidEmail(email)) {
      setErrors((prev) => ({ ...prev, email: '올바른 이메일 형식이 아닙니다.' }));
      return;
    }
    setErrors((prev) => ({ ...prev, email: undefined }));
    setIsEmailChecked(true);

    // TODO: API로 중복 확인
    const isDuplicate = false; // 임시 로직
    if (isDuplicate) {
      setEmailAvailable(false);
    } else {
      setEmailAvailable(true);
    }
  };

  const validate = () => {
    const newErrors: typeof errors = {};

    if (!isEmailChecked && !email) {
      // 이메일 미입력 시 자동 생성
    } else if (!isEmailChecked || !emailAvailable) {
      newErrors.email = '아이디 확인이 필요합니다.';
    }

    if (!password) newErrors.password = '비밀번호를 입력해주세요.';
    else if (!isValidPassword(password))
      newErrors.password = '영문(대소문자), 숫자, 특수문자 조합으로 10~20자 사용 가능합니다.';

    if (!confirmPassword) newErrors.confirmPassword = '비밀번호를 재입력해주세요.';
    else if (password !== confirmPassword) newErrors.confirmPassword = '비밀번호가 일치하지 않습니다.';

    if (!name) newErrors.name = '관리자 이름을 작성해주세요.';

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validate()) {
      const generatedEmail = email || `admin${Date.now()}@autogen.com`;
      alert('관리자가 생성되었습니다.');
      onCreate({ email: generatedEmail, name, role });
      onClose();
      resetForm();
    }
  };

  const resetForm = () => {
    setEmail('');
    setIsEmailChecked(false);
    setEmailAvailable(null);
    setPassword('');
    setConfirmPassword('');
    setName('');
    setRole('관리자');
    setErrors({});
  };

  if (!isOpen) return null;

  const ErrorMessage = ({ message }: { message: string }) => (
    <div className="flex items-center mt-1 text-red-500 text-sm">
      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <small className="p-error block">{message}</small>
    </div>
  );

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
      <div className="bg-white rounded-lg w-full max-w-md p-6 relative">
        <button className="absolute top-3 right-3 text-gray-500 hover:text-black" onClick={onClose}>
          <XMarkIcon className="h-6 w-6" />
        </button>

        <h2 className="text-xl font-bold mb-6">관리자 생성</h2>

        <form onSubmit={handleSubmit} className="space-y-5">
          {/* 관리자 아이디 */}
          <div>
            <div className="p-inputgroup flex gap-4 items-center border-b-2 border-gray-300 focus-within:border-blue-500 transition-all">
              <span className="p-inputgroup-addon bg-white px-0 py-2">
                <UserIcon className="h-5 w-5 text-gray-400" />
              </span>
              <InputText
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="관리자 아이디(이메일)"
                className={errors.email ? 'p-invalid' : ''}
                pt={{
                  root: {
                    className: `!w-full !py-2.5 !text-base focus:shadow-none !border-none ${errors.email ? 'p-invalid' : ''}`,
                  },
                }}
              />
              <Button
                label="확인"
                type="button"
                onClick={handleCheckEmail}
                pt={{
                  root: {
                    className:
                      '!bg-white !text-blue-500 !border !border-blue-500 !rounded-md !px-4 !py-2 !h-auto !text-sm !font-semibold !shadow-none !whitespace-nowrap !min-w-fit',
                  },
                  label: { className: 'whitespace-nowrap' },
                }}
              />
            </div>
            {isEmailChecked && emailAvailable && <p className="mt-1 text-green-500 text-sm">사용 가능한 아이디입니다.</p>}
            {isEmailChecked && emailAvailable === false && <ErrorMessage message="이미 사용 중인 아이디입니다." />}
            {errors.email && <ErrorMessage message={errors.email} />}
          </div>

          {/* 비밀번호 */}
          <div>
            <div className="p-inputgroup flex gap-4 items-center border-b-2 border-gray-300 focus-within:border-blue-500 transition-all">
              <span className="p-inputgroup-addon bg-white px-0 py-2">
                <LockClosedIcon className="h-5 w-5 text-gray-400" />
              </span>
              <Password
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="비밀번호를 입력해주세요"
                feedback={false}
                toggleMask
                className={`w-full ${errors.password ? 'p-invalid' : ''}`}
                pt={{
                  root: { className: 'flex-grow' },
                  input: {
                    className: `!w-full !py-2.5 !text-base focus:shadow-none !border-none ${errors.password ? 'p-invalid' : ''}`,
                  },
                  showIcon: { className: 'ml-auto mr-1 p-0' },
                  hideIcon: { className: 'ml-auto mr-1 p-0' },
                }}
              />
            </div>
            {errors.password && <ErrorMessage message={errors.password} />}
          </div>

          {/* 비밀번호 재입력 */}
          <div>
            <div className="p-inputgroup flex gap-4 items-center border-b-2 border-gray-300 focus-within:border-blue-500 transition-all">
              <span className="p-inputgroup-addon bg-white px-0 py-2">
                <LockClosedIcon className="h-5 w-5 text-gray-400" />
              </span>
              <Password
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="비밀번호를 재입력해주세요"
                feedback={false}
                toggleMask
                className={errors.confirmPassword ? 'p-invalid' : ''}
                pt={{
                  root: { className: 'flex-grow' },
                  input: {
                    className: `!w-full !py-2.5 !text-base focus:shadow-none !border-none ${errors.confirmPassword ? 'p-invalid' : ''}`,
                  },
                  showIcon: { className: 'ml-auto mr-1 p-0' },
                  hideIcon: { className: 'ml-auto mr-1 p-0' },
                }}
              />
            </div>
            {errors.confirmPassword && <ErrorMessage message={errors.confirmPassword} />}
          </div>

          {/* 이름 */}
          <div>
            <InputText
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="관리자 이름을 작성해주세요"
              className="w-full border-b-2 border-gray-300 rounded-none focus:border-blue-500 py-2 focus:outline-none focus:ring-0"
            />
            {errors.name && <ErrorMessage message={errors.name} />}
          </div>

          {/* 권한 */}
          <div className='flex items-center gap-4'>
            <p className='w-1/3'>관리자 권한 선택</p>
            <Dropdown
              value={role}
              options={[{ label: '관리자', value: '관리자' }]}
              onChange={(e) => setRole(e.value)}
              placeholder="관리자 권한 선택"
              className="w-2/3 border"
            />
          </div>

          <div className='flex justify-center'>
            <small className='text-[#A3A2A2]'>관리자 아이디 미입력시 임의로 아이디가 만들어집니다.</small>
          </div>

          <Button
            type="submit"
            label="관리자 생성"
            pt={{
              root: {
                className:
                  'w-full mt-6 py-3 bg-[#FFC107] text-white rounded-md text-lg font-semibold shadow-md hover:bg-[#e6b000] transition-colors duration-200 !shadow-none !border-none',
              },
            }}
          />
        </form>
      </div>
    </div>
  );
}
