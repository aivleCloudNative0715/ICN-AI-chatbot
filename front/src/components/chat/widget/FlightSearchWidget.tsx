// src/components/chat/widget/FlightSearchWidget.tsx
'use client';

import React, { useState } from 'react';
import { MagnifyingGlassIcon } from '@heroicons/react/24/solid';
import { getFlightArrivals, getFlightDepartures } from '@/lib/api';

export default function FlightSearchWidget() {
  const [flightNumber, setFlightNumber] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSearch = async () => {
    if (!flightNumber.trim()) {
      alert('항공편 번호를 입력해주세요.');
      return;
    }
    setIsLoading(true);
    try {
      // 도착편과 출발편을 동시에 검색
      const [arrivals, departures] = await Promise.all([
        getFlightArrivals(flightNumber),
        getFlightDepartures(flightNumber)
      ]);

      if (arrivals.length > 0) {
        const flight = arrivals[0];
        alert(`[도착] ${flight.flightId}편 (${flight.airport} 출발)\n현황: ${flight.remark}\n예정: ${flight.scheduleDateTime}, 변경: ${flight.estimatedDateTime}`);
      } else if (departures.length > 0) {
        const flight = departures[0];
        alert(`[출발] ${flight.flightId}편 (${flight.airport} 도착)\n현황: ${flight.remark}\n예정: ${flight.scheduleDateTime}, 변경: ${flight.estimatedDateTime}`);
      } else {
        alert(`${flightNumber}편을 찾을 수 없습니다.`);
      }

    } catch (error) {
      console.error("Flight search failed:", error);
      alert('항공편 조회 중 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white p-3 rounded-lg shadow-sm border border-gray-200">
      <h3 className="font-bold text-gray-800 mb-2 text-base">✈️ 항공편 빠른 조회</h3>
      <div className="flex items-center space-x-2">
        <input
          type="text"
          value={flightNumber}
          onChange={(e) => setFlightNumber(e.target.value.toUpperCase())}
          placeholder="항공편 번호 (예: KE123)"
          className="w-full px-3 py-1.5 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
          disabled={isLoading}
        />
        <button
          onClick={handleSearch}
          className="p-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400"
          disabled={isLoading}
        >
          {isLoading ? '...' : <MagnifyingGlassIcon className="h-4 w-4" />}
        </button>
      </div>
    </div>
  );
}