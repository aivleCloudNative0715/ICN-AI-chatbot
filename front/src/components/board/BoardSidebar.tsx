// src/components/board/BoardSidebar.tsx
'use client';

import React from 'react';
import { PanelMenu } from 'primereact/panelmenu';
import { MenuItem } from 'primereact/menuitem'; // Import MenuItem type for better type inference
import { PostCategory, PostFilter } from '../../lib/types'; // Assuming you create this file

interface BoardSidebarProps {
  onClose?: () => void;
  isLoggedIn: boolean;
  onCategorySelect: (category: PostCategory, filter: PostFilter) => void;
}

export default function BoardSidebar({ onClose, isLoggedIn, onCategorySelect }: BoardSidebarProps) {

  const items: MenuItem[] = [ // Explicitly type items as MenuItem[]
    {
      label: '문의 사항',
      expanded: true,
      items: [
        {
          label: '전체 문의 사항',
          command: () => onCategorySelect('inquiry', 'all'),
        },
        {
          label: '내 문의 사항',
          command: () => onCategorySelect('inquiry', 'my'),
          disabled: !isLoggedIn, // Disable if not logged in
        },
      ],
    },
    {
      label: '건의 사항',
      expanded: true,
      items: [
        {
          label: '전체 건의 사항',
          command: () => onCategorySelect('suggestion', 'all'),
        },
        {
          label: '내 건의 사항',
          command: () => onCategorySelect('suggestion', 'my'),
          disabled: !isLoggedIn, // Disable if not logged in
        },
      ],
    },
  ];

  return (
    <div className="flex w-80 p-4 bg-board-primary">
      <PanelMenu
        model={items}
        className="w-full"
        multiple
        pt={{
          headerAction: { className: 'flex items-center w-full text-board-dark border border-board-dark rounded-t-md bg-board-light' },
          headerLabel: { className: 'font-bold' },
          menuitem: { className: 'mb-2 text-xs text-board-dark' },
          menuContent: { className: 'border-s border-e border-b border-board-dark bg-board-primary' }
        }}
      />
    </div>
  );
}