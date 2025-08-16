'use client';

import React from 'react';
import { Button } from 'primereact/button';
import { ArrowPathIcon, TrashIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { DocumentTextIcon } from '@heroicons/react/24/solid';
import { Tooltip } from 'primereact/tooltip';
import { useRouter } from 'next/navigation';
import ParkingStatusWidget from './widget/ParkingStatusWidget';
import CongestionWidget from './widget/CongestionWidget';

interface ChatSidebarProps {
  isLoggedIn: boolean;
  onClose?: () => void;
  onDeleteAccount: () => void;
  // ✨ 1. 부모로부터 받을 함수 prop 타입을 추가합니다.
  onClearChatHistory: () => void;
}

export default function ChatSidebar({ isLoggedIn, onClose, onDeleteAccount, onClearChatHistory }: ChatSidebarProps) {
  const router = useRouter();
  const boardLinkId = "board-link-id";

  const handleNavigateToBoard = () => {
    if (isLoggedIn) {
      router.push('/board');
      if (onClose) onClose();
    }
  };

  return (
    <div className="fixed top-0 left-0 h-full w-[400px] bg-blue-100 shadow-lg p-4 flex flex-col justify-between z-20 transition-transform duration-300 ease-in-out transform translate-x-0">
      <div className="flex-grow">
        <div className="flex justify-end items-center mb-6">
          <button onClick={onClose} className="p-2 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500">
            <XMarkIcon className="h-6 w-6 text-gray-700" />
          </button>
        </div>

        {/* 정보 위젯들을 배치하는 영역 */}
        <div className="space-y-4">
            <CongestionWidget />
            <ParkingStatusWidget />
            {/* 혼잡도 API를 찾으시면 여기에 <CongestionWidget /> 추가 */}
        </div>

      </div>

      <div>
        <div className="space-y-4">
          <Button
            label="대화 기록 초기화"
            icon={<ArrowPathIcon className="h-5 w-5 mr-2" />}
            className="w-full pl-4 pr-4 h-10 rounded-full border border-gray-400 bg-white text-gray-700 flex items-center justify-center text-base font-semibold hover:bg-gray-100"
            onClick={onClearChatHistory}
          />
          {isLoggedIn && (
            <Button
              label="계정 삭제"
              icon={<TrashIcon className="h-5 w-5 mr-2" />}
              className="w-full pl-4 pr-4 h-10 rounded-full border border-red-500 bg-red-500 text-white flex items-center justify-center text-base font-semibold hover:bg-red-600"
              onClick={onDeleteAccount}
            />
          )}
        </div>
      </div>

      <div className="mt-auto">
        <div
          id={boardLinkId}
          className={`flex items-center py-2 px-3 rounded-md transition-colors duration-200 ${
            isLoggedIn ? 'hover:bg-blue-200 cursor-pointer' : 'opacity-50 cursor-not-allowed'
          }`}
          onClick={handleNavigateToBoard}
        >
          <DocumentTextIcon className="h-6 w-6 mr-2 text-gray-700" />
          <span className="font-medium text-gray-700">문의/건의 페이지로 이동하기</span>
        </div>
        
        {!isLoggedIn && (
          <Tooltip target={`#${boardLinkId}`} position="bottom">
            회원가입 후 이용 가능합니다.
          </Tooltip>
        )}
      </div>
    </div>
  );
}
