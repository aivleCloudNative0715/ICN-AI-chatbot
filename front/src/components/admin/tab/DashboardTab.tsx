// src/components/admin/tab/DashboardTab.tsx
'use client';

import React from 'react';
import AdminContentBoard from './AdminContentBoard';
import { useQuery } from '@tanstack/react-query'; // useQuery 임포트
import { getInquiryCounts } from '@/lib/api'; // api 함수 임포트
import { useAuth } from '@/contexts/AuthContext'; // token 가져오기

interface DashboardTabProps {
  onSelectInquiry: (inquiry: any) => void;
}

export default function DashboardTab({ onSelectInquiry }: DashboardTabProps) {
  const { token } = useAuth();

  // --- API를 통해 문의 건수 데이터를 가져오는 useQuery ---
  const { data: countsData, isLoading } = useQuery({
    queryKey: ['inquiryCounts'],
    queryFn: () => {
      if (!token) return null;
      // 오늘 날짜를 기준으로 조회 (혹은 원하는 기간으로 설정)
      const end = new Date();
      const start = new Date();
      start.setDate(end.getDate() - 30); // 예: 최근 30일
      
      return getInquiryCounts(
        token, 
        start.toISOString().slice(0, 19), 
        end.toISOString().slice(0, 19)
      );
    },
    enabled: !!token,
  });

  const summaryCards = [
    { label: '전체', value: countsData?.total ?? 0 },
    { label: '문의 사항', value: 'N/A' }, // 카테고리별 집계는 API 추가 필요
    { label: '건의 사항', value: 'N/A' },
    { label: '미처리', value: countsData?.pending ?? 0 },
    { label: '완료', value: countsData?.resolved ?? 0 },
  ];

  const priorityCards = [
    { label: '높음', value: 80, color: '#FB8C00' },
    { label: '보통', value: 80, color: '#1E88E5' },
    { label: '낮음', value: 80, color: '#81C784' },
  ];

  return (
    <div className="p-4 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">대시보드</h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
        {summaryCards.map((card, index) => (
          <div
            key={index}
            className="p-4 border border-l-[8px] border-[#C5C5C5] rounded-lg bg-white"
          >
            <p className="text-lg font-semibold">{card.label}</p>
            {/* 로딩 중일 때 스켈레톤 UI 또는 로딩 아이콘 표시 */}
            {isLoading ? (
              <div className="h-8 bg-gray-200 rounded animate-pulse mt-1"></div>
            ) : (
              <p className="text-2xl font-bold text-end">{card.value}</p>
            )}
          </div>
        ))}
      </div>

      {/* 우선순위 정보 (이전과 동일) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        {priorityCards.map((card, index) => (
          <div
            key={index}
            className={`p-4 border border-l-[8px] rounded-lg`}
            style={{ borderColor: card.color }}
          >
            <p className="text-lg font-semibold">{card.label}</p>
            <p className="text-2xl font-bold text-end">{card.value}</p>
          </div>
        ))}
      </div>

      {/* AdminContentBoard에 onSelectInquiry prop 전달 */}
      <AdminContentBoard type="dashboard" onSelectInquiry={onSelectInquiry} />
    </div>
  );
}