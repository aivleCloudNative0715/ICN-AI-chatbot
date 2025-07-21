// src/components/chat/ChatBotScreen.tsx
'use client';

import React from 'react';
import Image from 'next/image';
import SearchInput from '@/components/common/SearchInput';
import { PaperAirplaneIcon } from '@heroicons/react/24/outline'; // 화살표 아이콘

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

  // 하단 SearchInput의 높이를 고려하여 padding-bottom을 설정 (예시: 80px 또는 p-20)
  // 실제 SearchInput의 높이에 따라 조절이 필요합니다.
  const paddingBottomClass = "pb-20"; // 대략적인 SearchInput 높이에 맞춰 여유 공간 확보

  return (
    // 전체 컨테이너를 relative로 설정하여 하단 absolute 요소의 기준점 제공
    // items-center 제거: 하단 텍스트박스가 중앙으로 오지 않도록 (width=full 이므로 영향 적음)
    <div className={`relative flex flex-col flex-1 h-full bg-blue-50 ${paddingBottomClass}`}>
      {/* 챗봇 아이콘 및 인사말 (채팅 기록이 없을 때만 표시)*/}
      {chatMessages.length === 0 && (
        // flex-grow 제거: 이 섹션이 모든 공간을 차지하지 않도록 하여 하단 요소가 밀리지 않게 함
        <div className="flex flex-col items-center justify-center w-full flex-grow"> {/* flex-grow 유지하여 가운데 정렬 유지 */}
          <Image
            src="/airplane-icon.png" // UI/UX 문서의 로고와 동일한 이미지를 사용하려면 public/images/airport_logo.png 경로가 더 적합할 수 있습니다.
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
              <PaperAirplaneIcon className="h-6 w-6" />
            </span>
            <input
              type="text"
              placeholder="편명 입력"
              className="flex-grow bg-transparent outline-none text-center"
            />
          </div>
        </div>
      )}

      {/* 하단 텍스트 박스 (SearchInput 재사용) */}
      {/* absolute 포지셔닝을 사용하여 화면 하단에 고정 */}
      <div className="absolute bottom-0 left-0 right-0 p-4 bg-blue-50 shadow-md">
        <SearchInput placeholder="무엇이든 물어보세요!" />
      </div>
    </div>
  );
}