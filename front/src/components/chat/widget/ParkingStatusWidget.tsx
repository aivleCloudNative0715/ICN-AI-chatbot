// src/components/ParkingStatusWidget.tsx

'use client';

import React, { useState, useEffect } from 'react';

// API로부터 받을 주차장 데이터의 타입을 정의합니다.
interface ParkingInfo {
  name: string;
  occupied: number;
  total: number;
}

export default function ParkingStatusWidget() {
  const [parkingData, setParkingData] = useState<ParkingInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchParkingData = async () => {
      setIsLoading(true);
      // TODO: 여기에 실제 '주차장 잔여 조회' API를 호출하는 로직을 구현하세요.
      // const response = await fetch('/api/parking');
      // const data = await response.json();
      // setParkingData(data);

      // --- 임시 데이터 ---
      const MOCK_DATA: ParkingInfo[] = [
        { name: '제1터미널 단기(P1)', occupied: 234, total: 500 },
        { name: '제1터미널 장기(P2)', occupied: 89, total: 300 },
        { name: '제2터미널 단기(P3)', occupied: 124, total: 400 },
        { name: '제2터미널 장기(P4)', occupied: 350, total: 600 },
      ];
      // 1초 후 데이터를 보여주어 로딩 효과를 시뮬레이션합니다.
      setTimeout(() => {
        setParkingData(MOCK_DATA);
        setIsLoading(false);
      }, 1000);
      // --- 임시 데이터 끝 ---
    };

    fetchParkingData();
    // 1분마다 데이터를 새로고침하도록 설정할 수 있습니다.
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
            const percentage = (lot.occupied / lot.total) * 100;
            return (
              <div key={lot.name}>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm font-medium text-gray-700">{lot.name}</span>
                  <span className={`text-sm font-semibold ${percentage > 90 ? 'text-red-600' : 'text-gray-600'}`}>
                    {lot.occupied} / {lot.total}
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