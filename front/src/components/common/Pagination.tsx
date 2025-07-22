// src/components/common/Pagination.tsx
'use client';

import React from 'react';
import { Button } from 'primereact/button';
import { ChevronLeftIcon, ChevronRightIcon } from '@heroicons/react/24/outline';

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}

export default function Pagination({ currentPage, totalPages, onPageChange }: PaginationProps) {
  const getPageNumbers = () => {
    const pages = [];
    const maxPagesToShow = 5; // 한 번에 보여줄 페이지 번호 개수
    let startPage = Math.max(1, currentPage - Math.floor(maxPagesToShow / 2));
    let endPage = Math.min(totalPages, startPage + maxPagesToShow - 1);

    if (endPage - startPage + 1 < maxPagesToShow) {
      startPage = Math.max(1, endPage - maxPagesToShow + 1);
    }

    for (let i = startPage; i <= endPage; i++) {
      pages.push(i);
    }
    return pages;
  };

  const pages = getPageNumbers();

  return (
    <div className="flex items-center justify-center space-x-2 mt-8">
      <Button
        icon={<ChevronLeftIcon className="h-5 w-5" />}
        className="p-button-text p-button-sm !text-gray-600 hover:!bg-gray-100"
        onClick={() => onPageChange(currentPage - 1)}
        disabled={currentPage === 1}
        pt={{ root: { className: '!min-w-fit !p-2' } }}
      />

      {pages.map((page) => (
        <Button
          key={page}
          label={String(page)}
          className={`p-button-text p-button-sm ${
            page === currentPage ? '!bg-blue-500 !text-white' : '!text-gray-700 hover:!bg-gray-100'
          }`}
          onClick={() => onPageChange(page)}
          pt={{ root: { className: '!min-w-fit !px-3 !py-1' } }}
        />
      ))}

      <Button
        icon={<ChevronRightIcon className="h-5 w-5" />}
        className="p-button-text p-button-sm !text-gray-600 hover:!bg-gray-100"
        onClick={() => onPageChange(currentPage + 1)}
        disabled={currentPage === totalPages}
        pt={{ root: { className: '!min-w-fit !p-2' } }}
      />
    </div>
  );
}