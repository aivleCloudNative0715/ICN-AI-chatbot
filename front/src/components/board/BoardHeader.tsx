// src/components/board/BoardHeader.tsx
'use client';

import React from 'react';
import { Button } from 'primereact/button';
import { ArrowLeftOnRectangleIcon, ChatBubbleLeftRightIcon } from '@heroicons/react/24/outline'; // 아이콘 임포트

interface BoardHeaderProps {
  onGoToChat: () => void;
  onLogout: () => void;
}

export default function BoardHeader({ onGoToChat, onLogout }: BoardHeaderProps) {
  return (
    <header className="flex justify-end p-4 bg-white border-b border-gray-200 shadow-sm">
      <Button
        label="채팅으로 돌아가기"
        icon={<ChatBubbleLeftRightIcon className="h-5 w-5 mr-2" />}
        className="p-button-text p-button-sm !text-gray-600 hover:!bg-gray-100"
        onClick={onGoToChat}
        pt={{ root: { className: '!min-w-fit !px-3 !py-1' } }}
      />
      <Button
        label="로그아웃"
        icon={<ArrowLeftOnRectangleIcon className="h-5 w-5 mr-2" />}
        className="p-button-text p-button-sm !text-gray-600 hover:!bg-gray-100 ml-4"
        onClick={onLogout}
        pt={{ root: { className: '!min-w-fit !px-3 !py-1' } }}
      />
    </header>
  );
}