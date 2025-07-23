// src/components/common/FloatingActionButton.tsx
'use client';

import React from 'react';
import { Button } from 'primereact/button';
import { PlusIcon } from '@heroicons/react/24/outline';

interface FloatingActionButtonProps {
  onClick: () => void;
  label?: string;
}

export default function FloatingActionButton({ onClick, label }: FloatingActionButtonProps) {
  return (
    <Button
      icon={<PlusIcon className="h-6 w-6" />}
      label={label}
      className="fixed bottom-8 right-8 p-button-rounded p-button-success shadow-lg"
      onClick={onClick}
      pt={{
        root: {
          className: 'w-14 h-14 border bg-board-dark rounded-full flex items-center justify-center text-white text-xl shadow-lg hover:bg-board-primary hover:border-board-dark hover:text-board-dark transition-colors duration-200 !p-0'
        },
        icon: {
          className: 'h-6 w-6 text-white'
        },
        label: {
          className: label ? 'ml-2 hidden sm:block' : 'hidden' // 라벨이 있을 경우에만 표시
        }
      }}
    />
  );
}