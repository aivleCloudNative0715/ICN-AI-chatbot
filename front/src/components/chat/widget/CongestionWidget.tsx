// src/components/CongestionWidget.tsx

'use client';

import React, { useState, useEffect } from 'react';

// API 응답 데이터 중 필요한 부분의 타입을 정의합니다.
interface CongestionData {
  adate: string; // 날짜
  atime: string; // 시간대 (HHMM)
  t1sum1: string; // T1 입국장 합계
  t1sum2: string; // T1 출국장 합계
  t2sum1: string; // T2 입국장 합계
  t2sum2: string; // T2 출국장 합계
}

export default function CongestionWidget() {
  const [congestion, setCongestion] = useState<CongestionData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchCongestionData = async () => {
      setIsLoading(true);
      // TODO: 여기에 실제 '승객예고' API를 호출하는 로직을 구현하세요.
      // const response = await fetch('/api/congestion?date=today'); 
      // const data = await response.json();
      
      // --- 임시 데이터 (API 응답 예시) ---
      // 현재 시간에 맞춰 데이터를 필터링하거나 API에서 현재 시간대 데이터만 요청해야 합니다.
      const now = new Date();
      const currentHour = now.getHours();
      const MOCK_DATA: CongestionData = {
          adate: "20250816",
          atime: `${String(currentHour).padStart(2, '0')}00`,
          t1sum1: "1850", // T1 입국
          t1sum2: "3200", // T1 출국
          t2sum1: "980",  // T2 입국
          t2sum2: "2100", // T2 출국
      };
      // --- 임시 데이터 끝 ---
      
      setCongestion(MOCK_DATA);
      setIsLoading(false);
    };

    fetchCongestionData();
    // 10분마다 데이터를 새로고침하도록 설정할 수 있습니다.
    const interval = setInterval(fetchCongestionData, 600000); 
    return () => clearInterval(interval);
  }, []);

  // 승객 수에 따라 혼잡도 정보(텍스트, 색상)를 반환하는 함수
  const getCongestionInfo = (passengerCount: number) => {
    if (passengerCount <= 1000) return { level: '원활', color: 'text-green-600', bgColor: 'bg-green-100' };
    if (passengerCount <= 2500) return { level: '보통', color: 'text-yellow-600', bgColor: 'bg-yellow-100' };
    if (passengerCount <= 4000) return { level: '혼잡', color: 'text-orange-600', bgColor: 'bg-orange-100' };
    return { level: '매우 혼잡', color: 'text-red-600', bgColor: 'bg-red-100' };
  };

  if (isLoading) {
    return <div className="bg-white p-3 rounded-lg shadow-sm border border-gray-200 text-center">혼잡도 정보 로딩 중...</div>;
  }
  
  if (!congestion) {
      return <div className="bg-white p-3 rounded-lg shadow-sm border border-gray-200 text-center">혼잡도 정보를 불러올 수 없습니다.</div>;
  }

  const areas = [
      { name: 'T1 출국장', count: parseInt(congestion.t1sum2) },
      { name: 'T1 입국장', count: parseInt(congestion.t1sum1) },
      { name: 'T2 출국장', count: parseInt(congestion.t2sum2) },
      { name: 'T2 입국장', count: parseInt(congestion.t2sum1) },
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
            )
        })}
      </div>
      <p className="text-right text-xs text-gray-500 mt-2">{congestion.atime.substring(0,2)}:00 기준</p>
    </div>
  );
}