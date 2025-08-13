'use client';

import React, { useCallback, useEffect, useState } from 'react';
import BoardSidebar from '../../components/board/BoardSidebar';
import InquiryList from '../../components/board/InquiryList';
import { InquiryDto, PostCategory, PostFilter } from '../../lib/types';
import { deleteInquiry, getAllInquiries, getMyInquiries } from '../../lib/api';
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';
import { useRouter } from 'next/router';

export default function BoardPage() {
  const router = useRouter(); // router 훅 사용
  const [currentCategory, setCurrentCategory] = useState<PostCategory>('inquiry');
  const [currentFilter, setCurrentFilter] = useState<PostFilter>('all');
  const [searchTerm, setSearchTerm] = useState<string>('');

  // 1. 상태 변수 추가
  const [inquiries, setInquiries] = useState<InquiryDto[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // TODO: 실제 로그인한 사용자의 ID (Auth Context에서 가져와야 함)
  const currentUserId = 'user123';
  const isLoggedIn = true;

  // 2. API를 호출하는 함수
  const fetchInquiries = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      let response;
      if (currentFilter === 'my') {
        // '내 문의' 필터일 경우
        response = await getMyInquiries(currentUserId);
      } else {
        // '전체 문의' 필터일 경우
        response = await getAllInquiries();
      }
      setInquiries(response.content); // Page 객체의 content를 상태에 저장
    } catch (err) {
      setError(err instanceof Error ? err.message : '데이터 로딩 실패');
      setInquiries([]);
    } finally {
      setIsLoading(false);
    }
  }, [currentFilter, currentUserId]); // currentFilter가 바뀔 때마다 함수가 재생성됨

  // 3. 컴포넌트가 마운트되거나, 필터가 변경될 때 API 호출
  useEffect(() => {
    fetchInquiries();
  }, [fetchInquiries]);

  const handleCategorySelect = (category: PostCategory, filter: PostFilter) => {
    setCurrentCategory(category);
    setCurrentFilter(filter);
    // 필터가 변경되면 useEffect가 자동으로 API를 다시 호출
  };

  const handleSearch = () => {
    console.log('Searching for:', searchTerm);
  };

  // '내 문의' 필터가 활성화되었는지 여부를 결정합니다.
  const isMyList = currentFilter === 'my';

    // 삭제 처리 함수 추가
  const handleDelete = async (inquiryId: number) => {
    if (!window.confirm('정말로 이 문의를 삭제하시겠습니까?')) {
      return;
    }
    try {
      await deleteInquiry(inquiryId, currentUserId);
      alert('문의가 삭제되었습니다.');
      fetchInquiries(); // 목록을 다시 불러와 화면을 갱신
    } catch (err) {
      alert(err instanceof Error ? err.message : '삭제 중 오류가 발생했습니다.');
    }
  };

  return (
    <div className="flex">
      <BoardSidebar
        isLoggedIn={isLoggedIn}
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
        
         <InquiryList
          inquiries={inquiries}
          isLoading={isLoading}
          error={error}
          currentUserId={currentUserId}
          onDelete={handleDelete}
          onEdit={(inquiryId) => router.push(`/board/new?id=${inquiryId}`)}
        />
        
      </div>
    </div>
  );
}
