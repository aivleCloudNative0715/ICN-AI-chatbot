// src/components/chat/widget/CongestionWidget.tsx
'use client';

import React, { useState, useEffect } from 'react';
import { getPassengerForecast } from '@/lib/api'; // api.ts에서 함수 가져오기
import { PassengerForecast } from '@/lib/types'; // types.ts에서 타입 가져오기

export default function CongestionWidget() {
  const [forecast, setForecast] = useState<PassengerForecast | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchForecastData = async () => {
      try {
        setIsLoading(true);
        const data = await getPassengerForecast();
        
        // 현재 시간에 가장 가까운 예보를 찾습니다.
        const now = new Date();
        const currentHour = now.getHours();
        
        const currentForecast = data.find(item => {
          const [startHour] = item.atime.split('_').map(Number);
          return currentHour === startHour;
        });

        setForecast(currentForecast || data[0]); // 현재 시간 예보가 없으면 첫번째 데이터 사용
      } catch (error) {
        console.error("Failed to fetch passenger forecast:", error);
        setForecast(null);
      } finally {
        setIsLoading(false);
      }
    };

    fetchForecastData();
    // 10분마다 새로고침
    const interval = setInterval(fetchForecastData, 600000);
    return () => clearInterval(interval);
  }, []);

  const getCongestionInfo = (passengerCount: number) => {
    if (passengerCount <= 1000) return { level: '원활', color: 'text-green-600', bgColor: 'bg-green-100' };
    if (passengerCount <= 2500) return { level: '보통', color: 'text-yellow-600', bgColor: 'bg-yellow-100' };
    if (passengerCount <= 4000) return { level: '혼잡', color: 'text-orange-600', bgColor: 'bg-orange-100' };
    return { level: '매우 혼잡', color: 'text-red-600', bgColor: 'bg-red-100' };
  };

  if (isLoading) {
    return <div className="bg-white p-3 rounded-lg shadow-sm border border-gray-200 text-center">혼잡도 정보 로딩 중...</div>;
  }
  
  if (!forecast) {
    return <div className="bg-white p-3 rounded-lg shadow-sm border border-gray-200 text-center">혼잡도 정보를 불러올 수 없습니다.</div>;
  }

  const areas = [
    { name: 'T1 출국장', count: forecast.t1sumset2 },
    { name: 'T1 입국장', count: forecast.t1sumset1 },
    { name: 'T2 출국장', count: forecast.t2sumset2 },
    { name: 'T2 입국장', count: forecast.t2sumset1 },
  ];

  return (
    <div className="bg-white p-3 rounded-lg shadow-sm border border-gray-200">
      <h3 className="font-bold text-gray-800 mb-2 text-base">🕒 실시간 출/입국장 혼잡도</h3>
      <div className="grid grid-cols-2 gap-2">
        {areas.map(area => {
          const info = getCongestionInfo(area.count);
          return (
            <div key={area.name} className={`p-2 rounded-md ${info.bgColor}`}>
              <p className="font-semibold text-gray-700 text-sm">{area.name}</p>
              <p className={`font-bold text-lg ${info.color}`}>{info.level}</p>
            </div>
          );
        })}
      </div>
      <p className="text-right text-xs text-gray-500 mt-2">{forecast.atime.replace('_', ':')} 기준</p>
    </div>
  );
}