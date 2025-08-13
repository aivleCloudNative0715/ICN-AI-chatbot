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
import { InputText } from 'primereact/inputtext';
import { useDebounce } from '@/hooks/useDebounce'; 

export default function BoardPage() {
  const router = useRouter();
  const [currentCategory, setCurrentCategory] = useState<'INQUIRY' | 'SUGGESTION'>('INQUIRY');
  const [currentFilter, setCurrentFilter] = useState<PostFilter>('all');
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [inquiries, setInquiries] = useState<InquiryDto[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { isLoggedIn, user, token } = useAuth();

  // 디바운싱을 적용할 searchTerm을 useDebounce 훅에 넘겨줍니다. 
  // 사용자가 500ms(0.5초) 동안 타이핑을 멈추면 debouncedSearchTerm이 업데이트됩니다.
  const debouncedSearchTerm = useDebounce(searchTerm, 200);

  // DataTable 페이지네이션을 위한 상태 추가
  const [first, setFirst] = useState(0); // 첫 레코드의 인덱스
  const [rows, setRows] = useState(10); // 페이지당 행 수
  const [totalRecords, setTotalRecords] = useState(0); // 총 레코드 수

  const fetchInquiries = useCallback(async () => {
    if (!token) {
        setError('전체 목록을 보려면 로그인이 필요합니다.');
        setInquiries([]);
        return;
    }
    
    setIsLoading(true);
    setError(null);
    try {
      const page = first / rows; // 현재 페이지 번호 계산
      let response;
      if (currentFilter === 'my') {
        if (!user) throw new Error('인증 정보가 없습니다.');
        response = await getMyInquiries(currentCategory, token, page, rows, debouncedSearchTerm);
      } else {
        response = await getAllInquiries(currentCategory, token, page, rows, debouncedSearchTerm);
      }
      setInquiries(response.content);
      setTotalRecords(response.totalElements); // 총 레코드 수 업데이트
    } catch (err) {
      setError(err instanceof Error ? err.message : '데이터 로딩 실패');
      setInquiries([]);
    } finally {
      setIsLoading(false);
    }
  }, [token, currentCategory, currentFilter, user, first, rows, debouncedSearchTerm]);


  useEffect(() => {
    fetchInquiries();
  }, [fetchInquiries]);

  const onPageChange = (event: any) => {
    setFirst(event.first);
    setRows(event.rows);
  };

  const handleCategorySelect = (category: 'inquiry' | 'suggestion', filter: PostFilter) => {
    setCurrentCategory(category === 'inquiry' ? 'INQUIRY' : 'SUGGESTION');
    setCurrentFilter(filter);
  };

  return (
    <div className="flex flex-1">
      <BoardSidebar
        isLoggedIn={isLoggedIn}
        onCategorySelect={handleCategorySelect}
      />
      <div className="flex-1 p-6">
        <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold">{/*...*/}</h2>
            <span className="p-input-icon-left">
              <InputText 
                    value={searchTerm} 
                    // 사용자가 입력할 때마다 searchTerm state만 즉시 업데이트됩니다.
                    onChange={(e) => setSearchTerm(e.target.value)} 
                    placeholder="제목 또는 내용 검색" 
                    className="w-full border rounded-md py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline placeholder-gray-400"
              />
              <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
                <i className="pi pi-search text-gray-500" />
              </div>
            </span>
        </div>
        
        <InquiryList
          inquiries={inquiries}
          isLoading={isLoading}
          error={error}
          first={first}
          rows={rows}
          totalRecords={totalRecords}
          onPageChange={onPageChange}
        />
      </div>
    </div>
  );
}