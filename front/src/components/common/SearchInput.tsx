// src/components/common/SearchInput.tsx
'use client';

import React from 'react';
import { InputText } from 'primereact/inputtext';
import { Button } from 'primereact/button';
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';

interface SearchInputProps {
  onSearch?: (query: string) => void;
  placeholder?: string;
  initialQuery?: string;
}

export default function SearchInput({
  onSearch,
  placeholder = '무엇이든 물어보세요!', // 기본 플레이스홀더
  initialQuery = '',
}: SearchInputProps) {
  const [query, setQuery] = React.useState(initialQuery);

  const handleSearch = () => {
    if (onSearch) {
      onSearch(query);
      setQuery(''); // 검색 후 입력창 비우기
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div className="flex items-center w-full max-w-2xl mx-auto rounded-full shadow-md bg-white border border-gray-300 overflow-hidden">
      <InputText
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        className="flex-grow p-3 border-none focus:outline-none rounded-l-full"
        style={{ borderRadius: '9999px 0 0 9999px', paddingLeft: '1.5rem' }}
      />
      <Button
        icon={<MagnifyingGlassIcon className="h-6 w-6 text-gray-500" />}
        className="p-button-text p-button-icon-only rounded-r-full hover:bg-gray-100 p-3"
        onClick={handleSearch}
        style={{ borderRadius: '0 9999px 9999px 0' }}
      />
    </div>
  );
}