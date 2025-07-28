'use client';

import React from 'react';
import AdminContentBoard from './AdminContentBoard';

export default function DashboardTab() {
  // 데이터 정의
  const summaryCards = [
    { label: '전체', value: 80 },
    { label: '문의 사항', value: 80 },
    { label: '건의 사항', value: 80 },
    { label: '미처리', value: 80 },
    { label: '완료', value: 80 },
  ];

  const priorityCards = [
    { label: '높음', value: 80, color: '#FB8C00' },
    { label: '보통', value: 80, color: '#1E88E5' },
    { label: '낮음', value: 80, color: '#81C784' },
  ];

  return (
    <div className="p-4 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">대시보드</h2>

      {/* 대시보드 요약 정보 */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
        {summaryCards.map((card, index) => (
          <div
            key={index}
            className="p-4 border border-l-[8px] border-[#C5C5C5] rounded-lg bg-white"
          >
            <p className="text-lg font-semibold">{card.label}</p>
            <p className="text-2xl font-bold text-end">{card.value}</p>
          </div>
        ))}
      </div>

      {/* 우선순위 정보 */}
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

      <AdminContentBoard type="dashboard" />
    </div>
  );
}
