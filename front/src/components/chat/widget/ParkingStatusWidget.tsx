// src/components/chat/widget/ParkingStatusWidget.tsx
'use client';

import React from 'react';
import { ParkingInfo } from '@/lib/types';

interface ParkingStatusWidgetProps {
  data: ParkingInfo[];
  isLoading: boolean;
}

export default function ParkingStatusWidget({ data, isLoading }: ParkingStatusWidgetProps) {
  
  // 'YYYYMMDDHHmmss.SSS' 형식을 'HH:mm'으로 변환하는 함수
  const formatUpdateTime = (datetm: string) => {
    if (!datetm || datetm.length < 14) return '';
    const hour = datetm.substring(8, 10);
    const minute = datetm.substring(10, 12);
    return `${hour}:${minute}`;
  };
  
  const getBarColor = (percentage: number) => {
    if (percentage > 90) return 'bg-red-500';
    if (percentage > 70) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  // 데이터가 있을 경우 첫 번째 항목의 업데이트 시간을 가져옴
  const updateTime = !isLoading && data.length > 0 ? formatUpdateTime(data[0].datetm) : '';

  return (
    <div className="bg-white p-3 rounded-lg shadow-sm border border-gray-200">
      <div className="flex justify-between items-center mb-1">
        <h3 className="font-bold text-gray-800 text-base">🅿️ 실시간 주차 현황</h3>
        {updateTime && <span className="text-xs text-gray-500">조회시간: {updateTime}</span>}
      </div>
      {!isLoading && (
        <p className="text-gray-600 text-xs mb-2">
          실시간 주차 가능 대수이며, 지상 갓길 주차 및 불법 주차 차량도 포함될 수 있습니다.
        </p>
      )}
      {isLoading ? (
        <div className="text-center text-gray-500">주차 정보를 불러오는 중...</div>
      ) : (
        <div className="space-y-3">
          {data.map((lot) => {
            const occupied = parseInt(lot.parking);
            const total = parseInt(lot.parkingarea);
            // 최대치를 전체 주차 면수로 제한합니다.
            const displayOccupied = Math.min(occupied, total);
            const percentage = total > 0 ? (displayOccupied / total) * 100 : 0;

            return (
              <div key={lot.floor}>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm font-medium text-gray-700">{lot.floor}</span>
                  <span className={`text-sm font-semibold ${occupied > total ? 'text-red-700' : (percentage > 90 ? 'text-red-600' : 'text-gray-600')}`}>
                    {occupied.toLocaleString()} / {total.toLocaleString()}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                  <div
                    className={`${getBarColor(percentage)} h-2 rounded-full transition-all duration-500`}
                    style={{ width: `${percentage}%` }}
                  ></div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}