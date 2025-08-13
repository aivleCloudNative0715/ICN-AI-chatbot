'use client';

import React, { useState } from 'react';
import { PostCategory, PostFilter } from '../../lib/types';
import { ChevronDownIcon } from '@heroicons/react/24/solid';

interface BoardSidebarProps {
  onClose?: () => void;
  isLoggedIn: boolean;
  onCategorySelect: (category: PostCategory, filter: PostFilter) => void;
}

export default function BoardSidebar({ onClose, isLoggedIn, onCategorySelect }: BoardSidebarProps) {
  // ✅ 현재 활성화된(펼쳐진) 메뉴의 레이블을 저장하는 상태. '문의 사항'을 기본값으로 설정.
  const [activeMenu, setActiveMenu] = useState<string | null>('문의 사항');

  // 메뉴 데이터 구조는 그대로 사용합니다.
  const items = [
    {
      label: '문의 사항',
      items: [
        { label: '전체 문의 사항', category: 'inquiry', filter: 'all', disabled: false },
        { label: '내 문의 사항', category: 'inquiry', filter: 'my', disabled: !isLoggedIn },
      ],
    },
    {
      label: '건의 사항',
      items: [
        { label: '전체 건의 사항', category: 'suggestion', filter: 'all', disabled: false },
        { label: '내 건의 사항', category: 'suggestion', filter: 'my', disabled: !isLoggedIn },
      ],
    },
  ];

  // 메뉴 헤더를 클릭했을 때 호출될 함수
  const handleMenuToggle = (label: string) => {
    // 이미 열려있는 메뉴를 다시 클릭하면 닫고, 다른 메뉴를 클릭하면 그 메뉴를 엽니다.
    setActiveMenu(activeMenu === label ? null : label);
  };

  return (
    <div className="w-64 flex-shrink-0 p-4 bg-board-primary">
      <nav className="space-y-2">
        {items.map((menu) => (
          <div key={menu.label}>
            {/* 메뉴 헤더 (펼치기/접기 버튼) */}
            <button
              onClick={() => handleMenuToggle(menu.label)}
              // ✅ activeMenu 상태에 따라 rounded-t-md 와 rounded-md 클래스를 동적으로 적용합니다.
              className={`flex items-center justify-between w-full p-3 text-board-dark border border-board-dark bg-board-light hover:bg-gray-200 focus:outline-none ${
                activeMenu === menu.label ? 'rounded-t-md' : 'rounded-md'
              }`}
            >
              <span className="font-bold">{menu.label}</span>
              <ChevronDownIcon
                className={`h-5 w-5 transition-transform duration-200 ${
                  activeMenu === menu.label ? 'rotate-180' : ''
                }`}
              />
            </button>
            
            {/* 하위 메뉴 (펼쳐졌을 때만 보임) */}
            {activeMenu === menu.label && (
              <div className="py-2 pl-4 pr-2 border-s border-e border-b border-board-dark rounded-b-md">
                {menu.items.map((subItem) => (
                  <button
                    key={subItem.label}
                    onClick={() => onCategorySelect(subItem.category as PostCategory, subItem.filter as PostFilter)}
                    disabled={subItem.disabled}
                    className="w-full text-left p-2 text-sm rounded-md text-board-dark hover:bg-blue-100 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-transparent"
                  >
                    {subItem.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </nav>
    </div>
  );
}