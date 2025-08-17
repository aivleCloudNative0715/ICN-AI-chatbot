// src/components/chat/widget/CongestionWidget.tsx
'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { getPassengerForecast } from '@/lib/api';
import { PassengerForecast } from '@/lib/types';

interface CongestionWidgetProps {
  initialData: PassengerForecast[];
  isLoading: boolean;
}

const timeSlots = Array.from({ length: 24 }, (_, i) => {
  const start = String(i).padStart(2, '0');
  const end = String((i + 1) % 24).padStart(2, '0');
  return `${start}_${end === '00' ? '24' : end}`; // API 형식에 맞게 '23_24'로 표현
});

export default function CongestionWidget({ initialData, isLoading: initialLoading }: CongestionWidgetProps) {
  const [data, setData] = useState(initialData);
  const [isLoading, setIsLoading] = useState(initialLoading);
  const [selectedDay, setSelectedDay] = useState('0'); // '0': 오늘, '1': 내일
  const [selectedTime, setSelectedTime] = useState('');

  // 초기 데이터가 변경되면 내부 상태도 업데이트
  useEffect(() => {
    setData(initialData);
    setIsLoading(initialLoading);
    // 현재 시간에 맞는 시간대를 기본값으로 설정
    if (initialData.length > 0) {
      const currentHour = new Date().getHours();
      const defaultTimeSlot = timeSlots.find(slot => parseInt(slot.split('_')[0]) === currentHour);
      setSelectedTime(defaultTimeSlot || timeSlots[0]);
    }
  }, [initialData, initialLoading]);

  // '오늘'/'내일' 선택이 변경되면 API를 새로 호출
  const handleDayChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newDay = e.target.value;
    setSelectedDay(newDay);
    setIsLoading(true);
    try {
      // newDay 변수의 타입을 '0' | '1'로 단언해줍니다.
      const newData = await getPassengerForecast(newDay as '0' | '1');
      setData(newData);
    } catch (error) {
      console.error("Failed to fetch new forecast data:", error);
      setData([]);
    } finally {
      setIsLoading(false);
    }
  };

  // 선택된 시간대에 해당하는 예보 데이터를 찾음
  const selectedForecast = useMemo(() => 
    data.find(item => item.atime.startsWith(selectedTime.split('_')[0])), 
    [data, selectedTime]
  );
    
  const getCongestionInfo = (passengerCount: number) => {
    if (passengerCount <= 500) return { level: '원활', color: 'text-green-600' };
    if (passengerCount <= 1000) return { level: '보통', color: 'text-yellow-600' };
    if (passengerCount <= 1500) return { level: '혼잡', color: 'text-orange-600' };
    return { level: '매우 혼잡', color: 'text-red-600' };
  };

  const renderArea = (name: string, count: number | undefined) => {
    if (count === undefined) return null;
    const info = getCongestionInfo(count);
    return (
      <div className="flex justify-between items-center text-sm p-1.5 bg-gray-50 rounded">
        <span className="text-gray-600">{name}</span>
        <span className={`font-bold ${info.color}`}>{info.level} ({count.toLocaleString()})</span>
      </div>
    );
  };

  return (
    <div className="bg-white p-3 rounded-lg shadow-sm border border-gray-200">
      <h3 className="font-bold text-gray-800 mb-3 text-base">🕒 시간대별 승객 예고</h3>
      {!isLoading && (
        <p className="text-gray-600 text-xs mb-2">
          인천공항 제 1,2여객터미널의 입국심사, 입국장, 출국장에 출현할 시간대별 예상 승객 수 정보입니다. (오늘, 내일에 대한 예측 정보) 표기된 혼잡도는 자체적인 기준으로 판단한 결과이며, <span className="font-semibold text-red-500">예측 결과이므로 실제와 다를 수 있습니다.</span><br /><br />
          <span className="font-semibold">혼잡도 기준:</span>
          <span className="text-green-600 font-semibold"> 원활</span> (500명 이하),
          <span className="text-yellow-600 font-semibold"> 보통</span> (501 ~ 1000명),
          <span className="text-orange-600 font-semibold"> 혼잡</span> (1001 ~ 1500명),
          <span className="text-red-600 font-semibold"> 매우 혼잡</span> (1500명 초과)
        </p>
      )}
      <div className="flex space-x-2 mb-3">
        <select value={selectedDay} onChange={handleDayChange} className="flex-1 p-1.5 border rounded-md text-sm">
          <option value="0">오늘</option>
          <option value="1">내일</option>
        </select>
        <select value={selectedTime} onChange={(e) => setSelectedTime(e.target.value)} className="flex-1 p-1.5 border rounded-md text-sm">
          {timeSlots.map(slot => (
            <option key={slot} value={slot}>{`${slot.split('_')[0]}:00 ~ ${slot.split('_')[1] === '24' ? '24' : slot.split('_')[1]}:00`}</option>
          ))}
        </select>
      </div>

      {isLoading ? (
        <div className="text-center text-gray-500">데이터를 불러오는 중...</div>
      ) : selectedForecast ? (
        <div className="space-y-3">
          <div>
            <h4 className="font-semibold text-gray-700 text-sm mb-1">제1여객터미널</h4>
            <div className="grid grid-cols-1 gap-1">
              {renderArea('입국장(A,B)', selectedForecast.t1sum1)}
              {renderArea('입국장(E,F)', selectedForecast.t1sum2)}
              {renderArea('입국심사(C)', selectedForecast.t1sum3)}
              {renderArea('입국심사(D)', selectedForecast.t1sum4)}
              {renderArea('출국장(1,2)', selectedForecast.t1sum5)}
              {renderArea('출국장(3)', selectedForecast.t1sum6)}
              {renderArea('출국장(4)', selectedForecast.t1sum7)}
              {renderArea('출국장(5,6)', selectedForecast.t1sum8)}
            </div>
          </div>
          <div>
            <h4 className="font-semibold text-gray-700 text-sm mb-1">제2여객터미널</h4>
            <div className="grid grid-cols-1 gap-1">
              {renderArea('입국장(A)', selectedForecast.t2sum1)}
              {renderArea('입국장(B)', selectedForecast.t2sum2)}
              {renderArea('출국장(1)', selectedForecast.t2sum3)}
              {renderArea('출국장(2)', selectedForecast.t2sum4)}
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center text-gray-500">선택한 시간의 데이터가 없습니다.</div>
      )}
    </div>
  );
}