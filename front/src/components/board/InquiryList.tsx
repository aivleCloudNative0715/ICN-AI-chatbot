// src/components/board/InquiryList.tsx
'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { InputText } from 'primereact/inputtext';
import { Button } from 'primereact/button';
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';
import InquiryCard from './InquiryCard';
import Pagination from '@/components/common/Pagination';
import { InquiryListItem } from '@/lib/types';
import { CalendarIcon } from '@heroicons/react/24/solid'; // 날짜 정렬 아이콘 (예시)

interface InquiryListProps {
  category?: '문의' | '건의' | ''; // 필터링할 카테고리 (빈 문자열은 전체)
  isMyInquiries?: boolean; // 내 문의/건의 화면인지 여부
  currentUserId?: string; // 로그인한 user_id (내 문의/건의 필터링용)
}

export default function InquiryList({ category = '', isMyInquiries = false, currentUserId }: InquiryListProps) {
  const [inquiries, setInquiries] = useState<InquiryListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  const itemsPerPage = 5; // 한 페이지에 표시될 항목 수

  // 가상의 데이터 (실제 API 호출로 대체될 부분)
  const fetchInquiries = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // API-13-27037 (전체) 또는 API-13-27038 (내 문의) 호출 로직 시뮬레이션
      // 실제로는 fetch 또는 axios를 사용하여 백엔드 API를 호출해야 합니다.
      const allMockInquiries: InquiryListItem[] = [
        {
          inquiry_id: 'inq-001', user_id: 'user123', title: '배터리 충전 구역 문의', content: '공항 내에서 휴대폰 배터리를 충전할 수 있는 곳이 어디인가요? 콘센트 위치를 알고 싶습니다. 급하게 충전할 곳이 필요합니다. 감사합니다.', category: '문의', urgency: '보통', status: '답변처리 완료', created_at: '2025-07-20T10:00:00Z', updated_at: '2025-07-20T11:00:00Z', is_deleted: false, hasAnswer: true, answerContentPreview: '인천공항 내 무료 휴대폰 충전 서비스는 ...'
        },
        {
          inquiry_id: 'inq-002', user_id: 'user456', title: '공항 셔틀버스 시간표 건의', content: '공항 셔틀버스 배차 간격이 너무 길어서 불편합니다. 개선을 건의합니다. 특히 새벽 시간대에는 셔틀버스를 기다리는 시간이 너무 길어요.', category: '건의', urgency: '높음', status: '미처리', created_at: '2025-07-19T14:30:00Z', updated_at: '2025-07-19T14:30:00Z', is_deleted: false, hasAnswer: false
        },
        {
          inquiry_id: 'inq-003', user_id: 'user123', title: '환승객을 위한 라운지 이용 안내', content: '환승 시간이 긴데, 이용할 수 있는 라운지가 어디인지 궁금합니다. PP카드 사용이 가능한 라운지도 알려주세요. 자세한 안내 부탁드립니다.', category: '문의', urgency: '보통', status: '미처리', created_at: '2025-07-18T09:15:00Z', updated_at: '2025-07-18T09:15:00Z', is_deleted: false, hasAnswer: false
        },
        {
          inquiry_id: 'inq-004', user_id: 'user789', title: '흡연 구역 확충 요청', content: '흡연 구역이 너무 부족하여 불편합니다. 흡연자들을 위한 공간을 더 늘려주셨으면 좋겠습니다. 흡연 부스가 더 필요합니다.', category: '건의', urgency: '낮음', status: '답변처리 완료', created_at: '2025-07-17T16:45:00Z', updated_at: '2025-07-17T17:30:00Z', is_deleted: false, hasAnswer: true, answerContentPreview: '흡연 구역은 현재 법규에 따라 지정되어 있으며...'
        },
        {
          inquiry_id: 'inq-005', user_id: 'user123', title: '입국장 면세점 운영 시간 문의', content: '입국장 면세점 운영 시간이 어떻게 되는지 알려주세요. 새벽에 도착하는데 이용 가능한가요?', category: '문의', urgency: '높음', status: '미처리', created_at: '2025-07-16T11:20:00Z', updated_at: '2025-07-16T11:20:00Z', is_deleted: false, hasAnswer: false
        },
        {
          inquiry_id: 'inq-006', user_id: 'user456', title: '분실물 센터 연락처 문의', content: '공항에서 물건을 잃어버렸는데 분실물 센터 연락처와 위치를 알고 싶습니다.', category: '문의', urgency: '보통', status: '답변처리 완료', created_at: '2025-07-15T13:00:00Z', updated_at: '2025-07-15T14:00:00Z', is_deleted: false, hasAnswer: true, answerContentPreview: '분실물 센터는 터미널 1 지하 1층에 위치하며...'
        },
        {
          inquiry_id: 'inq-007', user_id: 'user123', title: '공항 와이파이 속도 개선 건의', content: '공항 내 무료 와이파이 속도가 너무 느려서 사용하기 어렵습니다. 개선이 필요합니다.', category: '건의', urgency: '보통', status: '미처리', created_at: '2025-07-14T10:00:00Z', updated_at: '2025-07-14T10:00:00Z', is_deleted: false, hasAnswer: false
        },
      ].filter(inq => !inq.is_deleted); // 논리적 삭제된 항목은 제외

      // 필터링 및 검색 로직
      let filteredInquiries = allMockInquiries;

      if (category) { // '문의' 또는 '건의' 카테고리 필터링
        filteredInquiries = filteredInquiries.filter(inq => inq.category === category);
      }

      if (isMyInquiries && currentUserId) { // '내 문의/건의' 필터링
        filteredInquiries = filteredInquiries.filter(inq => inq.user_id === currentUserId);
      }

      if (searchQuery) { // 제목, 내용, 답변 내용 검색
        const lowerCaseQuery = searchQuery.toLowerCase();
        filteredInquiries = filteredInquiries.filter(inq =>
          inq.title.toLowerCase().includes(lowerCaseQuery) ||
          inq.content.toLowerCase().includes(lowerCaseQuery) ||
          (inq.answerContentPreview && inq.answerContentPreview.toLowerCase().includes(lowerCaseQuery))
        );
      }

      // 날짜(created_at) 기준 내림차순 정렬
      filteredInquiries.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

      // 페이지네이션 적용
      const totalCount = filteredInquiries.length;
      const calculatedTotalPages = Math.ceil(totalCount / itemsPerPage);
      setTotalPages(calculatedTotalPages === 0 ? 1 : calculatedTotalPages); // 최소 1페이지

      const startIndex = (currentPage - 1) * itemsPerPage;
      const endIndex = startIndex + itemsPerPage;
      const paginatedInquiries = filteredInquiries.slice(startIndex, endIndex);

      setInquiries(paginatedInquiries);
    } catch (err) {
      setError('데이터를 불러오는 데 실패했습니다.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [category, isMyInquiries, currentUserId, currentPage, searchQuery]);

  useEffect(() => {
    fetchInquiries();
  }, [fetchInquiries]);

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
    setCurrentPage(1); // 검색 시 첫 페이지로 이동
  };

  // InquiryCard에서 전달받을 수정/삭제 핸들러
  const handleInquiryEdit = (inquiryId: string) => {
    console.log(`Edit inquiry: ${inquiryId}`);
    // API-13-27041 (문의/건의 수정) 호출 로직은 InquiryForm 페이지에서 처리될 예정
    // 여기서는 단순히 ID를 가지고 수정 페이지로 리다이렉트
  };

  const handleInquiryDelete = async (inquiryId: string) => {
    console.log(`Delete inquiry: ${inquiryId}`);
    // API-13-27042 (문의/건의 삭제) 호출 로직 시뮬레이션
    // 실제 백엔드 API 호출 후, 성공하면 fetchInquiries() 재호출
    setLoading(true);
    try {
      // await fetch(`/api/inquiries/${inquiryId}`, { method: 'DELETE' });
      alert('문의/건의가 삭제되었습니다.');
      fetchInquiries(); // 목록 갱신
    } catch (err) {
      alert('삭제 중 오류가 발생했습니다.');
      console.error(err);
      setLoading(false);
    }
  };


  if (loading) {
    return <div className="text-center py-10">데이터 로딩 중...</div>;
  }

  if (error) {
    return <div className="text-center py-10 text-red-500">{error}</div>;
  }

  return (
    <div className="flex-grow p-6 bg-blue-50 relative">
      <div className="flex items-center justify-between mb-6">
        <div className="relative flex items-center w-full max-w-lg mx-auto rounded-full shadow-md bg-white border border-gray-300 overflow-hidden">
          <InputText
            placeholder="제목, 내용, 답변 내용을 검색하세요."
            value={searchQuery}
            onChange={handleSearchChange}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                // 검색 실행 (이미 onChange에서 상태 업데이트되므로 별도 로직 불필요)
              }
            }}
            className="flex-grow p-3 border-none focus:outline-none rounded-l-full"
            style={{ borderRadius: '9999px 0 0 9999px', paddingLeft: '1.5rem' }}
          />
          <Button
            icon={<MagnifyingGlassIcon className="h-6 w-6 text-gray-500" />}
            className="p-button-text p-button-icon-only rounded-r-full hover:bg-gray-100 p-3"
            onClick={fetchInquiries} // 검색 버튼 클릭 시 데이터 다시 불러오기
            style={{ borderRadius: '0 9999px 9999px 0' }}
          />
        </div>
        {/* 정렬 기준은 UI에 없으므로 일단 생략 (날짜 기준 정렬은 fetchInquiries에 포함) */}
        {/* <Button
          icon={<CalendarIcon className="h-5 w-5 mr-2" />}
          label="최신순"
          className="p-button-outlined"
          onClick={() => alert('정렬 기능 예정')}
        /> */}
      </div>

      {inquiries.length === 0 ? (
        <div className="text-center text-gray-500 py-10">
          게시글이 없습니다.
        </div>
      ) : (
        <div className="space-y-4">
          {inquiries.map((inquiry) => (
            <InquiryCard
              key={inquiry.inquiry_id}
              inquiry={inquiry}
              isMyInquiries={isMyInquiries}
              onEdit={handleInquiryEdit}
              onDelete={handleInquiryDelete}
            />
          ))}
        </div>
      )}

      <Pagination
        currentPage={currentPage}
        totalPages={totalPages}
        onPageChange={handlePageChange}
      />
    </div>
  );
}