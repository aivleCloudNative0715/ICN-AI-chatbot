// src/components/common/Header.tsx
'use client';

import React from 'react';
import { Bars3Icon } from '@heroicons/react/24/outline';
import { Button } from 'primereact/button';
import SpeechBubble from './SpeechBubble';

interface HeaderProps {
  onLoginClick: () => void;
  onRegisterClick: () => void;
  onMenuClick?: () => void;
  isLoggedIn?: boolean;
  onLogoutClick?: () => void;
}

export default function Header({
  onLoginClick,
  onRegisterClick,
  onMenuClick,
  isLoggedIn = false,
  onLogoutClick,
}: HeaderProps) {
  return (
    <header className="fixed top-0 left-0 right-0 z-10 flex items-center justify-between p-4 bg-secondary-light h-16">
      <div className="flex items-center">
        <button
          onClick={onMenuClick}
          className="p-2 mr-2 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
        >
          <Bars3Icon className="h-6 w-6 text-neutral-dark-text" />
        </button>
        <span className="text-xl font-bold text-neutral-dark-text">인천공항 챗봇</span>
      </div>

      <div className="flex items-center space-x-2">
        {isLoggedIn ? (
          <Button
            label="로그아웃"
            className="
              bg-neutral-white text-primary border border-primary rounded-md px-3 py-1.5 text-sm
              hover:bg-primary hover:text-neutral-white hover:border-primary /* 호버 시 반전 */
              focus:ring-2 focus:ring-primary focus:ring-offset-2
            "
            onClick={onLogoutClick}
          />
        ) : (
          <div className='flex gap-4'>
            <div className="flex items-center gap-4">
              <SpeechBubble
                message="회원가입 시 채팅 기록이 최대 하루동안 유지됩니다."
                position="right"
                minWidth="200px"
                maxWidth="fit-content"
              />
              <Button
                label="회원가입"
                className="
                  bg-accent-yellow text-neutral-white border border-accent-yellow rounded-md px-3 py-1.5 text-sm
                  hover:bg-secondary-light hover:text-accent-yellow hover:border-accent-yellow hover: border-2
                  focus:ring-2 focus:ring-accent-yellow focus:ring-offset-2
                "
                onClick={onRegisterClick}
              />
            </div>
            <Button
              label="로그인"
              className="
                bg-secondary-light text-primary border-2 border-primary rounded-md px-3 py-1.5 text-sm
                hover:bg-primary hover:text-neutral-white hover:border-primary
                focus:ring-2 focus:ring-primary focus:ring-offset-2
              "
              onClick={onLoginClick}
            />
          </div>
        )}
      </div>
    </header>
  );
}