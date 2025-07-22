// src/components/board/BoardSidebar.tsx
'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ChevronRightIcon, ChevronDownIcon } from '@heroicons/react/24/outline';

interface BoardSidebarProps {
  onClose?: () => void; // 사이드바를 닫는 함수 (필요 시)
  isLoggedIn: boolean; // 로그인 상태 prop
}

export default function BoardSidebar({ onClose, isLoggedIn }: BoardSidebarProps) {
  const pathname = usePathname();
  const [openInquiry, setOpenInquiry] = React.useState(true);
  const [openSuggestion, setOpenSuggestion] = React.useState(true);

  return (
    <div className="flex flex-col h-full bg-blue-100 p-4 border-r border-gray-200 w-64">
      {/* 문의 사항 섹션 */}
      <div className="mb-4">
        <button
          className="flex items-center justify-between w-full py-2 px-3 text-lg font-semibold text-gray-800 rounded-md hover:bg-blue-200"
          onClick={() => setOpenInquiry(!openInquiry)}
        >
          문의 사항
          {openInquiry ? (
            <ChevronDownIcon className="h-5 w-5" />
          ) : (
            <ChevronRightIcon className="h-5 w-5" />
          )}
        </button>
        {openInquiry && (
          <div className="pl-4 mt-2 space-y-1">
            <Link href="/board" passHref>
              <span
                className={`block py-2 px-3 text-sm rounded-md cursor-pointer ${
                  pathname === '/board' ? 'bg-blue-200 text-blue-800 font-medium' : 'text-gray-700 hover:bg-blue-100'
                }`}
              >
                전체 문의 사항
              </span>
            </Link>
            {isLoggedIn && (
              <Link href="/board/my-inquiries" passHref>
                <span
                  className={`block py-2 px-3 text-sm rounded-md cursor-pointer ${
                    pathname === '/board/my-inquiries' ? 'bg-blue-200 text-blue-800 font-medium' : 'text-gray-700 hover:bg-blue-100'
                  }`}
                >
                  내 문의 사항
                </span>
              </Link>
            )}
          </div>
        )}
      </div>

      {/* 건의 사항 섹션 (문의 사항과 동일한 구조) */}
      <div className="mb-4">
        <button
          className="flex items-center justify-between w-full py-2 px-3 text-lg font-semibold text-gray-800 rounded-md hover:bg-blue-200"
          onClick={() => setOpenSuggestion(!openSuggestion)}
        >
          건의 사항
          {openSuggestion ? (
            <ChevronDownIcon className="h-5 w-5" />
          ) : (
            <ChevronRightIcon className="h-5 w-5" />
          )}
        </button>
        {openSuggestion && (
          <div className="pl-4 mt-2 space-y-1">
            <Link href="/board?category=건의" passHref> {/* 건의 사항 전체는 쿼리 파람으로 구분 */}
              <span
                className={`block py-2 px-3 text-sm rounded-md cursor-pointer ${
                  pathname === '/board' && new URLSearchParams(window.location.search).get('category') === '건의'
                    ? 'bg-blue-200 text-blue-800 font-medium'
                    : 'text-gray-700 hover:bg-blue-100'
                }`}
              >
                전체 건의 사항
              </span>
            </Link>
            {isLoggedIn && (
              <Link href="/board/my-inquiries?category=건의" passHref> {/* 내 건의 사항도 쿼리 파람으로 구분 */}
                <span
                  className={`block py-2 px-3 text-sm rounded-md cursor-pointer ${
                    pathname === '/board/my-inquiries' && new URLSearchParams(window.location.search).get('category') === '건의'
                      ? 'bg-blue-200 text-blue-800 font-medium'
                      : 'text-gray-700 hover:bg-blue-100'
                  }`}
                >
                  내 건의 사항
                </span>
              </Link>
            )}
          </div>
        )}
      </div>

      {/* "채팅으로 돌아가기" 버튼은 Header에서 관리하거나 필요시 여기에 추가 */}
      {/* 현재 사이드바는 게시판 전용이므로 "채팅으로 돌아가기"는 Header에 더 적합 */}
    </div>
  );
}