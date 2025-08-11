'use client';

import React, { useState } from 'react';
import BoardSidebar from '../../components/board/BoardSidebar';
import InquiryList from '../../components/board/InquiryList';
import { PostCategory, PostFilter } from '../../lib/types'; // Import types
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';

export default function BoardPage() {
  const [currentCategory, setCurrentCategory] = useState<PostCategory>('inquiry');
  const [currentFilter, setCurrentFilter] = useState<PostFilter>('all');
  const [searchTerm, setSearchTerm] = useState<string>('');

  // Assume this is your logged-in user's ID
  const currentUser = { id: 'user123', isLoggedIn: true }; // Example user

  const handleCategorySelect = (category: PostCategory, filter: PostFilter) => {
    setCurrentCategory(category);
    setCurrentFilter(filter);
  };

  const handleSearch = () => {
    // This function can be used for future API-based search
    console.log('Searching for:', searchTerm);
  };

  // '내 문의' 필터가 활성화되었는지 여부를 결정합니다.
  const isMyList = currentFilter === 'my';

  return (
    <div className="flex">
      <BoardSidebar
        isLoggedIn={currentUser.isLoggedIn}
        onCategorySelect={handleCategorySelect}
      />
      <div className="flex-1 p-4">
        {/* Search Bar */}
        <div className="relative flex items-center justify-center w-full max-w-sm px-4 py-3 border-b-2 border-board-dark text-gray-700 placeholder-gray-400 focus-within:border-blue-500 transition-all duration-300 mx-auto">
          <span className="mr-2 text-board-dark">
            <MagnifyingGlassIcon className="h-6 w-6" />
          </span>
          <input
            type="text"
            placeholder="검색어를 입력하세요..."
            className="flex-grow bg-transparent outline-none text-center"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                handleSearch();
              }
            }}
          />
        </div>
        
        <h2>{currentCategory === 'inquiry' ? '문의 사항' : '건의 사항'} ({currentFilter === 'all' ? '전체' : '내'})</h2>
        
        <InquiryList isMyInquiries={isMyList} currentUserId={currentUser.id} />
        
      </div>
    </div>
  );
}
