// src/components/admin/AdminHeader.tsx
'use client';

import { useRouter } from 'next/navigation';

interface AdminHeaderProps {
  onLogoutClick: () => void;
}

export default function AdminHeader({ onLogoutClick }: AdminHeaderProps) {
  const router = useRouter();

  return (
    <header className="flex justify-between items-center p-4 bg-white text-black shadow-md">
      {/* 관리자 대시보드 타이틀 */}
      <div className="flex items-center">
        <h1 className="text-2xl font-bold">관리자 대시보드</h1>
      </div>

      {/* 로그아웃 버튼 */}
      <div>
        <button
          onClick={onLogoutClick}
          className="px-4 py-2 bg-white border border-black rounded-md hover:bg-black hover:text-white transition duration-300"
        >
          로그아웃
        </button>
      </div>
    </header>
  );
}