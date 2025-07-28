// src/components/common/SearchInput.tsx
'use client';

import React from 'react';
import { InputText } from 'primereact/inputtext';
import { Button } from 'primereact/button';
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';

interface SearchInputProps {
  onSearch?: (query: string) => void;
  placeholder?: string;
  value?: string; // 입력 값 제어
  onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void; 
  initialQuery?: string;
  onSend?: () => void;
}

export default function SearchInput({ placeholder = '검색어를 입력하세요', value, onChange, onSend }: SearchInputProps) {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && onSend) {
      onSend();
    }
  };

  return (
    <div className="flex items-center w-full max-w-2xl mx-auto rounded-full shadow-md bg-white border border-gray-300 overflow-hidden">
      <InputText
        placeholder={placeholder}
        value={value}
        onChange={onChange} 
        onKeyDown={handleKeyDown}
        className="flex-grow p-3 border-none focus:outline-none rounded-l-full"
        style={{ borderRadius: '9999px 0 0 9999px', paddingLeft: '1.5rem' }}
      />
      <Button
        icon={<MagnifyingGlassIcon className="h-6 w-6 text-gray-500" />}
        className="p-button-text p-button-icon-only rounded-r-full hover:bg-gray-100 p-3"
        onClick={onSend}
        style={{ borderRadius: '0 9999px 9999px 0' }}
      />
    </div>
  );
}