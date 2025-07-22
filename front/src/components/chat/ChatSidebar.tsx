// src/components/chat/ChatSidebar.tsx
'use client';

import React, { useRef, useState, useEffect } from 'react'; // useEffect 추가
import { Button } from 'primereact/button';
import { ArrowPathIcon, TrashIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { DocumentTextIcon } from '@heroicons/react/24/solid';
import { Tooltip } from 'primereact/tooltip';
import { useRouter } from 'next/navigation';// useRouter 임포트 

interface ChatSidebarProps {
  isLoggedIn: boolean;
  onClose?: () => void; // 사이드바 닫기 핸들러
}

export default function ChatSidebar({ isLoggedIn, onClose }: ChatSidebarProps) {
  const router = useRouter();// useRouter 훅 초기화 
  // const boardLinkRef = useRef<HTMLDivElement>(null); // 이제 직접적인 ref 대신 ID를 사용할 것임

  // 대화 기록 초기화 핸들러 (API-09-25033)
  const handleClearChatHistory = () => {
    if (confirm('대화 기록을 초기화하시겠습니까?')) {
      alert('대화 기록이 초기화되었습니다.'); // TODO: 실제 API 호출
    }
  };

  // 계정 삭제 핸들러 (API-09-26034)
  const handleDeleteAccount = () => {
    if (confirm('계정을 삭제하시겠습니까? 이 작업은 되돌릴 수 없습니다.')) {
      alert('계정이 삭제되었습니다.'); // TODO: 실제 API 호출
    }
  };

  // 툴팁 대상이 될 요소의 고유 ID를 정의
  const boardLinkId = "board-link-id";

  // 게시판으로 이동 핸들러
  const handleNavigateToBoard = () => {
   if (isLoggedIn) { // 로그인 상태일 때만 이동 
      router.push('/board');// /board 경로로 이동 
      if (onClose) {
        onClose(); // 사이드바 닫기 (선택 사항)
      }
    }
  };

  return (
    <div className="fixed top-0 left-0 h-full w- bg-blue-100 shadow-lg p-4 flex flex-col justify-between z-20 transition-transform duration-300 ease-in-out transform translate-x-0">
      {/* width를 48로 재조정, 이전 제안에서 w-124로 잘못 표기된 것 같음 (Tailwind에 w-124 없음) */}
      <div className="flex justify-end items-center mb-6">
        <button onClick={onClose} className="p-2 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500">
          <XMarkIcon className="h-6 w-6 text-gray-700" />
        </button>
      </div>

      <div>
        <div className="space-y-4">
          <Button
            label="대화 기록 초기화"
            icon={<ArrowPathIcon className="h-5 w-5 mr-2" />}
            className="pl-4 pr-4 h-10 rounded-full border border-[#C50000] bg-secondary-DEFAULT text-[#C50000] flex items-center justify-center text-base font-semibold"
            onClick={handleClearChatHistory}
          />
          {isLoggedIn && (
            <Button
              label="계정 삭제"
              icon={<TrashIcon className="h-5 w-5 mr-2" />}
              className="pl-4 pr-4 h-10 rounded-full border border-[#C50000] bg-[#C50000] text-[#FFFFFF] flex items-center justify-center text-base font-semibold"
              onClick={handleDeleteAccount}
            />
          )}
        </div>
      </div>

      <div className="mt-auto">
        <div
          id={boardLinkId} // 고유 ID 할당
          // ref={boardLinkRef} // ref는 이제 필요 없음
          className={`flex items-center py-2 px-3 rounded-md transition-colors duration-200 ${
            isLoggedIn ? 'hover:bg-blue-200 cursor-pointer' : 'opacity-50 cursor-not-allowed'
          }`}
          // alert 대신 게시판 이동 함수 호출
          onClick={handleNavigateToBoard}
        >
          <DocumentTextIcon className="h-6 w-6 mr-2 text-gray-700" />
          <span className="font-medium text-gray-700">문의/건의 페이지로 이동하기</span>
        </div>
        
        {/* 툴팁 (로그인하지 않았을 때만 표시) */}
        {/* target에 요소의 ID 문자열을 전달 */}
        {!isLoggedIn && (
          <Tooltip target={`#${boardLinkId}`} position="bottom">
            회원가입 후 이용 가능합니다.
          </Tooltip>
        )}
      </div>
    </div>
  );
}