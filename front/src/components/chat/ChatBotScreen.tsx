// src/components/chat/ChatBotScreen.tsx
'use client';

import React from 'react';
import Image from 'next/image';
import SearchInput from '@/components/common/SearchInput';
import { ChevronRightIcon } from '@heroicons/react/24/outline'; // 화살표 아이콘

interface ChatBotScreenProps {
  isLoggedIn: boolean; // 로그인 상태 prop
  onLoginStatusChange: (status: boolean) => void; // 로그인 상태 변경 핸들러
  onSidebarToggle: () => void; // 사이드바 토글 핸들러
}

export default function ChatBotScreen({
  isLoggedIn,
  onLoginStatusChange,
  onSidebarToggle,
}: ChatBotScreenProps) {
  // TODO: 실제 채팅 메시지 목록 상태 관리 (나중에 구현)
  const chatMessages: any[] = []; // 현재는 빈 배열로 초기화

  return (
    <div className="flex flex-col flex-1 h-full items-center justify-between p-6 bg-blue-50">
      {/* 챗봇 아이콘 및 인사말 (채팅 기록이 없을 때만 표시)*/}
      {chatMessages.length === 0 && (
        <div className="flex flex-col items-center justify-center flex-grow">
          <Image
            src="/airplane-icon.png"
            alt="Airplane Icon"
            width={150}
            height={150}
            className="mb-6"
          />
          <h1 className="text-2xl font-semibold text-gray-800 mb-2 text-center">
            인천공항 AI 챗봇 서비스입니다! 궁금한 점을 물어봐주세요!
          </h1>
          <p className="text-gray-600 mb-8 text-center">
            편명 입력 시 더 자세한 답변이 가능합니다.
          </p>

          {/* 편명 입력 텍스트 박스*/}
          <div className="relative flex items-center justify-center w-full max-w-sm px-4 py-3 border-b-2 border-gray-300 text-gray-700 placeholder-gray-400 focus-within:border-blue-500 transition-all duration-300">
            <span className="mr-2 text-gray-500">
              <ChevronRightIcon className="h-5 w-5" />
            </span>
            <input
              type="text"
              placeholder="편명 입력"
              className="flex-grow bg-transparent outline-none text-center"
            />
          </div>
        </div>
      )}

      {/* 하단 텍스트 박스 (SearchInput 재사용)*/}
      <div className="w-full sticky bottom-0 p-4 bg-blue-50">
        <SearchInput placeholder="무엇이든 물어보세요!" />
      </div>
    </div>
  );
}