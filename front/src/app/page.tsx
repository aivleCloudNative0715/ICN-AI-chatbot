// src/app/page.tsx
'use client'; // 클라이언트 컴포넌트로 지정

import ChatBotScreen from '@/components/chat/ChatBotScreen';
import ChatSidebar from '@/components/chat/ChatSidebar';
import { useState } from 'react';

export default function HomePage() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false); // 임시 로그인 상태 (실제 앱에서는 Context API 등으로 전역 관리)

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  // 이 함수는 AuthModal에서 로그인 성공 시 호출될 수 있도록 설계.
  // 실제로는 Context API 등을 통해 전역 로그인 상태를 업데이트하는 것이 일반적.
  const handleLoginStatusChange = (status: boolean) => {
    setIsLoggedIn(status);
  };

  return (
    <div className="relative flex flex-1 w-full h-full">
      {/* 사이드바 (열려 있을 때만 렌더링) */}
      {isSidebarOpen && <ChatSidebar isLoggedIn={isLoggedIn} />}

      {/* 챗봇 메인 화면 */}
      <ChatBotScreen
        isLoggedIn={isLoggedIn}
        onLoginStatusChange={handleLoginStatusChange}
        onSidebarToggle={toggleSidebar}
      />
    </div>
  );
}