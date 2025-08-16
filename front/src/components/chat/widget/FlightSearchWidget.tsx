// src/components/FlightSearchWidget.tsx

'use client';

import React, { useState } from 'react';
import { MagnifyingGlassIcon } from '@heroicons/react/24/solid';

export default function FlightSearchWidget() {
  const [flightNumber, setFlightNumber] = useState('');

  const handleSearch = () => {
    if (!flightNumber.trim()) {
      alert('항공편 번호를 입력해주세요.');
      return;
    }
    // TODO: 챗봇에게 메시지를 보내거나, API를 호출하여 결과를 모달로 보여주는 로직 구현
    alert(`항공편 ${flightNumber} 조회를 요청합니다.`);
    // 예: onSearch(flightNumber); // 부모 컴포넌트로 검색어 전달
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
        />
        <button
          onClick={handleSearch}
          className="p-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        >
          <MagnifyingGlassIcon className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}