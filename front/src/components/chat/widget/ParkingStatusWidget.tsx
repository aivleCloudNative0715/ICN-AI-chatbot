// src/components/chat/widget/ParkingStatusWidget.tsx
'use client';

import React, { useState, useEffect } from 'react';
import { getParkingStatus } from '@/lib/api';
import { ParkingInfo } from '@/lib/types';

// 각 주차장의 전체 주차 가능 대수를 상수로 관리합니다.
const PARKING_CAPACITY: { [key: string]: number } = {
  'T1 단기주차장': 2200, // 예시 값
  'T1 장기주차장': 15000, // 예시 값
  'T2 단기주차장': 1200, // 예시 값
  'T2 장기주차장': 8000, // 예시 값
};

export default function ParkingStatusWidget() {
  const [parkingData, setParkingData] = useState<ParkingInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchParkingData = async () => {
      try {
        setIsLoading(true);
        const data = await getParkingStatus();
        setParkingData(data);
      } catch (error) {
        console.error("Failed to fetch parking status:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchParkingData();
    const interval = setInterval(fetchParkingData, 60000);
    return () => clearInterval(interval);
  }, []);

  const getBarColor = (percentage: number) => {
    if (percentage > 90) return 'bg-red-500';
    if (percentage > 70) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  return (
    <div className="bg-white p-3 rounded-lg shadow-sm border border-gray-200">
      <h3 className="font-bold text-gray-800 mb-3 text-base">🅿️ 실시간 주차 현황</h3>
      {isLoading ? (
        <div className="text-center text-gray-500">데이터를 불러오는 중...</div>
      ) : (
        <div className="space-y-3">
          {parkingData.map((lot) => {
            const total = PARKING_CAPACITY[lot.floor] || parseInt(lot.parking);
            const occupied = parseInt(lot.parking);
            const percentage = total > 0 ? (occupied / total) * 100 : 0;
            
            return (
              <div key={lot.floor}>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm font-medium text-gray-700">{lot.floor}</span>
                  <span className={`text-sm font-semibold ${percentage > 90 ? 'text-red-600' : 'text-gray-600'}`}>
                    {occupied} / {total}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
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