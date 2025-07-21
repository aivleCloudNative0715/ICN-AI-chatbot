// src/components/chat/ChatSidebar.tsx
'use client';

import React from 'react';
import { Button } from 'primereact/button';
import { ArrowPathIcon, TrashIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { DocumentTextIcon } from '@heroicons/react/24/solid';

interface ChatSidebarProps {
  isLoggedIn: boolean;
  onClose?: () => void; // 사이드바 닫기 핸들러
}

export default function ChatSidebar({ isLoggedIn, onClose }: ChatSidebarProps) {
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

  return (
    <div className="fixed top-0 left-0 h-full w-64 bg-blue-100 shadow-lg p-6 flex flex-col justify-between z-20 transition-transform duration-300 ease-in-out transform translate-x-0">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-lg font-bold text-gray-800">설정</h2>
        <button onClick={onClose} className="p-2 rounded-full hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500">
          <XMarkIcon className="h-6 w-6 text-gray-700" />
        </button>
      </div>

      <div>
        <div className="space-y-4">
          <Button
            label="대화 기록 초기화"
            icon={<ArrowPathIcon className="h-5 w-5 mr-2" />}
            className="p-button-danger p-button-outlined w-full"
            onClick={handleClearChatHistory}
          />
          {isLoggedIn && (
            <Button
              label="계정 삭제"
              icon={<TrashIcon className="h-5 w-5 mr-2" />}
              className="p-button-danger w-full"
              onClick={handleDeleteAccount}
            />
          )}
        </div>
      </div>

      <div className="mt-auto">
        {/* 게시판 이동 하이퍼링크 (로그인 여부에 따라 활성화/비활성화) */}
        <div
          className={`flex items-center text-blue-700 py-2 px-3 rounded-md transition-colors duration-200 ${
            isLoggedIn ? 'hover:bg-blue-200 cursor-pointer' : 'opacity-50 cursor-not-allowed'
          }`}
          onClick={isLoggedIn ? () => alert('게시판 페이지로 이동!') : undefined}
        >
          <DocumentTextIcon className="h-6 w-6 mr-2" />
          <span className="font-medium">문의/건의 페이지로 이동하기</span>
        </div>
        {!isLoggedIn && (
          <p className="text-xs text-red-500 mt-2">회원가입 후 이용 가능합니다.</p>
        )}
      </div>
    </div>
  );
}