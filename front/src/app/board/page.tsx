// src/app/board/page.tsx
'use client';

import React, { useCallback, useEffect, useState } from 'react';
import BoardSidebar from '@/components/board/BoardSidebar';
import InquiryList from '@/components/board/InquiryList';
import { InquiryDto, PostCategory, PostFilter } from '@/lib/types';
import { deleteInquiry, getAllInquiries, getMyInquiries } from '@/lib/api';
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';
import { useAuth } from '@/contexts/AuthContext';
import { useRouter } from 'next/navigation';

export default function BoardPage() {
  const router = useRouter();
  const [currentCategory, setCurrentCategory] = useState<PostCategory>('inquiry');
  const [currentFilter, setCurrentFilter] = useState<PostFilter>('all');
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [inquiries, setInquiries] = useState<InquiryDto[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { isLoggedIn, user, token } = useAuth();

  const fetchInquiries = useCallback(async () => {
    if (!token) { // 토큰이 없으면 일부 API 호출 불가
      if(currentFilter === 'my') {
        setError('로그인이 필요합니다.');
        setInquiries([]);
        setIsLoading(false);
        return;
      }
    }
    
    setIsLoading(true);
    setError(null);
    try {
      let response;
      if (currentFilter === 'my') {
        if (!user || !token) throw new Error('인증 정보가 없습니다.');
        response = await getMyInquiries(user.userId, token);
      } else {
        response = await getAllInquiries();
      }
      setInquiries(response.content);
    } catch (err) {
      setError(err instanceof Error ? err.message : '데이터 로딩 실패');
      setInquiries([]);
    } finally {
      setIsLoading(false);
    }
  }, [currentFilter, user, token]);

  useEffect(() => {
    fetchInquiries();
  }, [fetchInquiries]);

  const handleCategorySelect = (category: PostCategory, filter: PostFilter) => {
    setCurrentCategory(category);
    setCurrentFilter(filter);
  };
  
  const handleSearch = () => { console.log('Searching for:', searchTerm); };

  const handleDelete = async (inquiryId: number) => {
    if (!user || !token) return alert('로그인이 필요합니다.');
    if (window.confirm('정말로 이 문의를 삭제하시겠습니까?')) {
      try {
        await deleteInquiry(inquiryId, user.userId, token);
        alert('문의가 삭제되었습니다.');
        fetchInquiries();
      } catch (err) {
        alert(err instanceof Error ? err.message : '삭제 중 오류가 발생했습니다.');
      }
    }
  };

  return (
    <div className="flex flex-1">
      <BoardSidebar
        isLoggedIn={isLoggedIn}
        onCategorySelect={handleCategorySelect}
      />
      <div className="flex-1 p-4">
        <div className="relative ..."> {/* 검색 바 */} </div>
        
        <h2>{currentCategory === 'inquiry' ? '문의 사항' : '건의 사항'} ({currentFilter === 'all' ? '전체' : '내 문의'})</h2>
        
        <InquiryList
          inquiries={inquiries}
          isLoading={isLoading}
          error={error}
          currentUserId={user?.userId || ''} 
          onDelete={handleDelete}
          onEdit={(inquiryId) => router.push(`/board/new?id=${inquiryId}`)}
        />
      </div>
    </div>
  );
}