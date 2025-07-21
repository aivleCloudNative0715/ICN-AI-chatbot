// src/components/common/Modal.tsx
'use client';

import React from 'react';
import { Dialog } from 'primereact/dialog';
import { XMarkIcon } from '@heroicons/react/24/outline';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}

export default function Modal({ isOpen, onClose, title, children }: ModalProps) {
  return (
    <Dialog
      header={
        <div className="flex items-center justify-between w-full">
          <span className="text-xl font-bold">{title}</span>
          <button onClick={onClose} className="p-2 rounded-full hover:bg-gray-100 focus:outline-none">
            <XMarkIcon className="h-6 w-6 text-gray-700" />
          </button>
        </div>
      }
      visible={isOpen}
      onHide={onClose}
      modal
      draggable={false}
      resizable={false}
      blockScroll
      className="w-11/12 md:w-1/3"
    >
      {children}
    </Dialog>
  );
}