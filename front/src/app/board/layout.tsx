// src/app/(board)/layout.tsx
'use client';

import React from 'react';
import { usePathname, useRouter } from 'next/navigation';
import BoardSidebar from '@/components/board/BoardSidebar';
import FloatingActionButton from '@/components/common/FloatingActionButton';
import { Button } from 'primereact/button';
import { ArrowLeftOnRectangleIcon, ChatBubbleLeftRightIcon } from '@heroicons/react/24/outline';

interface BoardLayoutProps {
  children: React.ReactNode;
}

export default function BoardLayout({ children }: BoardLayoutProps) {
  const router = useRouter();
  const pathname = usePathname();
  // TODO: 실제 isLoggedIn 상태는 Context API 또는 전역 상태 관리 훅에서 가져와야 함
  // 현재는 임시로 true로 설정하거나, 부모 layout에서 props로 전달받도록 할 수 있음
  const isLoggedIn = true; // 임시 로그인 상태. 실제는 Auth Context 등에서 가져올 것.

  const handleGoToChat = () => {
    router.push('/'); // 메인 챗봇 화면으로 이동
  };

  const handleLogout = () => {
    // TODO: 실제 API 호출 및 토큰 삭제 로직 (API-08-20030)
    localStorage.removeItem('jwt_token'); // 예시: 로컬 스토리지에서 토큰 삭제
    // setIsLoggedIn(false); // 전역 로그인 상태 업데이트
    alert('로그아웃되었습니다.');
    router.push('/'); // 로그아웃 후 메인 챗봇 화면으로 이동
  };

  const handleNewInquiryClick = () => {
    if (isLoggedIn) {
      router.push('/board/new');
    } else {
      alert('로그인 후 이용 가능합니다.');
      // TODO: 로그인 모달 열기 등의 액션
    }
  };

  const isNewPage = pathname === '/board/new';

  return (
    <div className="flex flex-1 h-full bg-blue-50">
      {/* 사이드바 */}
      <BoardSidebar isLoggedIn={isLoggedIn} />

      {/* 메인 콘텐츠 영역 */}
      <div className="flex flex-col flex-1 h-full">
        {/* 상단 액션 버튼 */}
        <div className="flex justify-end p-4 bg-white border-b border-gray-200 shadow-sm">
          <Button
            label="채팅으로 돌아가기"
            icon={<ChatBubbleLeftRightIcon className="h-5 w-5 mr-2" />}
            className="p-button-text p-button-sm !text-gray-600 hover:!bg-gray-100"
            onClick={handleGoToChat}
            pt={{ root: { className: '!min-w-fit !px-3 !py-1' } }}
          />
          <Button
            label="로그아웃"
            icon={<ArrowLeftOnRectangleIcon className="h-5 w-5 mr-2" />}
            className="p-button-text p-button-sm !text-gray-600 hover:!bg-gray-100 ml-4"
            onClick={handleLogout}
            pt={{ root: { className: '!min-w-fit !px-3 !py-1' } }}
          />
        </div>

        {/* 페이지 콘텐츠 */}
        <main className="flex-grow flex flex-col overflow-y-auto">
          {children}
        </main>

        {/* 새 문의/건의 추가 Floating Action Button */}
        {!isNewPage && isLoggedIn && ( // 새 작성 페이지가 아닐 때만, 로그인 시 표시
          <FloatingActionButton onClick={handleNewInquiryClick} label="새 문의/건의" />
        )}
      </div>
    </div>
  );
}