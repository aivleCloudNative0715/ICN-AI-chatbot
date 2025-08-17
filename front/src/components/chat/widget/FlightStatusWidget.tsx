// src/components/chat/widget/FlightStatusWidget.tsx
'use client';

import React from 'react';
import { FlightArrival, FlightDeparture } from '@/lib/types';
import { ClockIcon, PaperAirplaneIcon, ArrowDownTrayIcon, ArrowUpTrayIcon } from '@heroicons/react/24/outline';

interface FlightStatusWidgetProps {
  arrivals: FlightArrival[];
  departures: FlightDeparture[];
  isLoading: boolean;
}

export default function FlightStatusWidget({ arrivals, departures, isLoading }: FlightStatusWidgetProps) {

  // 현재 시간을 "HH:mm" 형식으로 포맷하는 함수
  const getCurrentTime = () => {
    return new Date().toLocaleTimeString('ko-KR', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    });
  };

  const renderFlightItem = (flight: FlightArrival | FlightDeparture, type: 'arrival' | 'departure') => {
    // 예정 시간과 변경 시간을 HH:mm 형식으로 추출
    const scheduledTime = `${flight.scheduleDateTime.substring(8, 10)}:${flight.scheduleDateTime.substring(10, 12)}`;
    const estimatedTime = `${flight.estimatedDateTime.substring(8, 10)}:${flight.estimatedDateTime.substring(10, 12)}`;
    
    // 두 시간이 다른지 확인 (null 또는 빈 문자열 체크 포함)
    const isTimeChanged = flight.estimatedDateTime && flight.scheduleDateTime !== flight.estimatedDateTime;

    return (
      <div key={flight.flightId + flight.scheduleDateTime} className="flex items-center space-x-3 text-sm p-1">
        <div className="flex-shrink-0">
          <PaperAirplaneIcon className={`h-5 w-5 ${type === 'arrival' ? 'text-blue-500' : 'text-red-500'}`} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex justify-between items-center">
            <p className="font-bold text-gray-800 truncate">{flight.flightId}</p>
            <p className="text-xs text-gray-500 flex-shrink-0 ml-2">{type === 'arrival' ? `← ${flight.airport}` : `→ ${flight.airport}`}</p>
          </div>
          <div className="flex justify-between items-center text-xs text-gray-600">
            <span>{flight.remark}</span>
            
            {/* ▼▼▼▼▼ 시간 표시 로직 수정 ▼▼▼▼▼ */}
            <div className="flex items-center space-x-1">
              <ClockIcon className="h-3 w-3" />
              {isTimeChanged ? (
                <>
                  <s className="text-gray-400">{scheduledTime}</s>
                  <span className="font-bold text-red-500">→ {estimatedTime}</span>
                </>
              ) : (
                <span>{scheduledTime}</span>
              )}
            </div>
            {/* ▲▲▲▲▲ 시간 표시 로직 수정 ▲▲▲▲▲ */}

          </div>
        </div>
      </div>
    );
  };
  return (
    <div className="bg-white p-3 rounded-lg shadow-sm border border-gray-200">
      <div className="flex justify-between items-center mb-1">
        <h3 className="font-bold text-gray-800 text-base">✈️ 실시간 운항 정보</h3>
        {!isLoading && <span className="text-xs text-gray-500">조회시간: {getCurrentTime()}</span>}
      </div>
      {!isLoading && (
        <p className="text-gray-600 text-xs mb-2">
          현재 시간 기준으로 이후 5개의 항공편만 표시됩니다.
        </p>
      )}
      {isLoading ? (
        <div className="text-center text-gray-500 h-40 flex items-center justify-center">항공편 정보를 불러오는 중...</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-bold text-gray-700 mb-2 flex items-center text-sm"><ArrowDownTrayIcon className="h-5 w-5 mr-2 text-blue-600"/>도착</h4>
            <div className="space-y-1">
              {arrivals.length > 0 ? arrivals.slice(0, 5).map(flight => renderFlightItem(flight, 'arrival')) : <p className="text-xs text-gray-500">정보 없음</p>}
            </div>
          </div>
          <div>
            <h4 className="font-bold text-gray-700 mb-2 flex items-center text-sm"><ArrowUpTrayIcon className="h-5 w-5 mr-2 text-red-600"/>출발</h4>
            <div className="space-y-1">
              {departures.length > 0 ? departures.slice(0, 5).map(flight => renderFlightItem(flight, 'departure')) : <p className="text-xs text-gray-500">정보 없음</p>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}