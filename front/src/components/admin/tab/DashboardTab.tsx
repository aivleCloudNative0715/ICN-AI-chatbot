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
      
      return getInquiryCounts(token);
    },
    enabled: !!token,
  });

  const summaryCards = [
    { label: '전체', value: countsData?.total ?? 0 },
    { label: '문의 사항', value: countsData?.inquiry ?? 0 }, // API 데이터로 교체
    { label: '건의 사항', value: countsData?.suggestion ?? 0 }, // API 데이터로 교체
    { label: '미처리', value: countsData?.pending ?? 0 },
    { label: '완료', value: countsData?.resolved ?? 0 },
  ];

  const priorityCards = [
    { label: '높음', value: countsData?.high ?? 0, color: '#FB8C00' }, // API 데이터로 교체
    { label: '보통', value: countsData?.medium ?? 0, color: '#1E88E5' }, // API 데이터로 교체
    { label: '낮음', value: countsData?.low ?? 0, color: '#81C784' }, // API 데이터로 교체
  ];

  return (
    <div className="p-4 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">대시보드</h2>

      {/* 요약 카드 */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
        {summaryCards.map((card, index) => (
          <div
            key={index}
            className="p-4 border border-l-[8px] border-[#C5C5C5] rounded-lg bg-white"
          >
            <p className="text-lg font-semibold">{card.label}</p>
            {isLoading ? (
              <div className="h-8 bg-gray-200 rounded animate-pulse mt-1"></div>
            ) : (
              <p className="text-2xl font-bold text-end">{card.value}</p>
            )}
          </div>
        ))}
      </div>

      {/* 우선순위 카드 */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        {priorityCards.map((card, index) => (
          <div
            key={index}
            className={`p-4 border border-l-[8px] rounded-lg`}
            style={{ borderColor: card.color }}
          >
            <p className="text-lg font-semibold">{card.label}</p>
            {/* 우선순위 카드에도 로딩 상태를 적용합니다. */}
            {isLoading ? (
              <div className="h-8 bg-gray-200 rounded animate-pulse mt-1"></div>
            ) : (
              <p className="text-2xl font-bold text-end">{card.value}</p>
            )}
          </div>
        ))}
      </div>

      {/* AdminContentBoard에 onSelectInquiry prop 전달 */}
      <AdminContentBoard type="dashboard" onSelectInquiry={onSelectInquiry} />
    </div>
  );
}