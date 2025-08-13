// src/components/board/BoardHeader.tsx
'use client';

import React from 'react';
import { Button } from 'primereact/button';
import { ArrowLeftOnRectangleIcon, ChatBubbleLeftRightIcon } from '@heroicons/react/24/outline'; // 아이콘 임포트
import { useRouter } from 'next/navigation';

interface BoardHeaderProps {
  onGoToChat: () => void;
  onLogout: () => void;
}

export default function BoardHeader({ onGoToChat, onLogout }: BoardHeaderProps) {
  const router = useRouter();

  const handleLogoutClick = () => {
    onLogout();
    router.push('/');
  };

  return (
    <header className="flex justify-end gap-4 p-4 bg-board-primary">
      <Button
        label="채팅으로 돌아가기"
        className="
          bg-board-primary text-baord-dark border-2 border-board-dark rounded-md px-3 py-1.5 text-sm
          hover:bg-borad-dark hover:text-white hover:bg-board-dark
          focus:ring-2 focus:ring-accent-yellow focus:ring-offset-2
        "
        onClick={onGoToChat}
        pt={{ root: { className: '!min-w-fit !px-3 !py-1' } }}
      />
      <Button
        label="로그아웃"
        className="
          bg-board-dark text-white border-2 border-board-dark rounded-md px-3 py-1.5 text-sm
          hover:bg-board-primary hover:text-board-dark hover:border-board-dark hover:border-2
          focus:ring-2 focus:ring-primary focus:ring-offset-2
        "
        onClick={handleLogoutClick}
        pt={{ root: { className: '!min-w-fit !px-3 !py-1' } }}
      />
    </header>
  );
}