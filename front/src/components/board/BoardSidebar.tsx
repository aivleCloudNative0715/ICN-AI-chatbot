'use client';

import React from 'react';
import { PanelMenu } from 'primereact/panelmenu';

interface BoardSidebarProps {
  onClose?: () => void;
  isLoggedIn: boolean;
}

export default function BoardSidebar({ onClose, isLoggedIn }: BoardSidebarProps) {

      const items = [
        {
            label: '문의 사항',
            expanded: true,
            items: [
                {
                    label: '전체 문의 사항',
                },
                {
                    label: '내 문의 사항',
                }
            ]
        },
        {
            label: '건의 사항',
            expanded: true,
            items: [
                {
                    label: '전체 건의 사항',
                },
                {
                    label: '내 건의 사항',
                },
            ]
        },
    ];

  return (
    <div className="flex w-80 p-4">
      <PanelMenu
        model={items}
        className="w-full"
        multiple
        pt={{
          headerAction: {className: 'flex items-center w-full text-board-dark border border-board-dark rounded-t-md bg-board-light'},
          headerLabel: {className: 'font-bold'},
          menuitem: { className: 'mb-2 text-xs text-board-dark' },
          menuContent: {className: 'border border-board-dark bg-board-primary'}
        }}
      />
    </div>
  );
}
