'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Button } from 'primereact/button';
import { ArrowPathIcon, TrashIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { DocumentTextIcon } from '@heroicons/react/24/solid';
import { Tooltip } from 'primereact/tooltip';
import { useRouter } from 'next/navigation';

// 위젯 컴포넌트들을 import 합니다.
import ParkingStatusWidget from './widget/ParkingStatusWidget';
import CongestionWidget from './widget/CongestionWidget';
import FlightStatusWidget from './widget/FlightStatusWidget';
import WeatherWidget from './widget/WeatherWidget';


// API 호출 함수와 타입을 import 합니다.
import { getParkingStatus, getPassengerForecast, getFlightArrivals, getFlightDepartures, getLatestTemperature } from '@/lib/api';
import { ParkingInfo, PassengerForecast, FlightArrival, FlightDeparture, TemperatureInfo   } from '@/lib/types';


interface ChatSidebarProps {
  isLoggedIn: boolean;
  onClose?: () => void;
  onDeleteAccount: () => void;
  onClearChatHistory: () => void;
}

export default function ChatSidebar({ isLoggedIn, onClose, onDeleteAccount, onClearChatHistory }: ChatSidebarProps) {
  const router = useRouter();
  const boardLinkId = "board-link-id";

  const [isLoading, setIsLoading] = useState(true);
  const [parkingData, setParkingData] = useState<ParkingInfo[]>([]);
  const [forecastData, setForecastData] = useState<PassengerForecast[]>([]);
  const [arrivalFlights, setArrivalFlights] = useState<FlightArrival[]>([]);
  const [departureFlights, setDepartureFlights] = useState<FlightDeparture[]>([]);
  const [weatherData, setWeatherData] = useState<TemperatureInfo | null>(null);

  const fetchAllData = useCallback(async () => {
    setIsLoading(true);
    try {
      // Promise.all을 사용해 모든 API를 병렬로 호출합니다.
      const [parking, forecast, arrivals, departures, weather] = await Promise.all([
        getParkingStatus(),
        getPassengerForecast('0'), // 기본값으로 '오늘' 데이터 호출
        getFlightArrivals(),
        getFlightDepartures(),
        getLatestTemperature(),
      ]);

      setParkingData(parking);
      setForecastData(forecast);
      setArrivalFlights(arrivals);
      setDepartureFlights(departures);
      setWeatherData(weather);

    } catch (error) {
      console.error("Failed to fetch airport data:", error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAllData();
  }, [fetchAllData]);

  const handleNavigateToBoard = () => {
    if (isLoggedIn) {
      router.push('/board');
      if (onClose) onClose();
    }
  };

  return (
    <div className="fixed top-0 left-0 h-full w-[600px] bg-blue-100 shadow-lg p-4 flex flex-col justify-between z-20 transition-transform duration-300 ease-in-out transform translate-x-0">
      <div className="flex-grow min-h-0">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-bold text-blue-800">실시간 공항 정보</h2>
          <div className="flex items-center space-x-2">
            {/* 새로고침 버튼 */}
            <button onClick={fetchAllData} className="p-2 rounded-full hover:bg-blue-200" title="새로고침">
              <ArrowPathIcon className={`h-6 w-6 text-gray-700 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
            <button onClick={onClose} className="p-2 rounded-full hover:bg-blue-200">
              <XMarkIcon className="h-6 w-6 text-gray-700" />
            </button>
          </div>
        </div>

        {/* 위젯 영역 (스크롤 가능하도록 수정) */}
        <div className="space-y-4 overflow-y-auto pr-2" style={{maxHeight: 'calc(100vh - 250px)'}}>
          {/* 각 위젯에 필요한 데이터와 로딩 상태를 props로 전달합니다. */}
          <WeatherWidget weather={weatherData} isLoading={isLoading} />
          <FlightStatusWidget arrivals={arrivalFlights} departures={departureFlights} isLoading={isLoading} />
          <CongestionWidget initialData={forecastData} isLoading={isLoading} />
          <ParkingStatusWidget data={parkingData} isLoading={isLoading} />
        </div>
      </div>
      
      {/* 하단 기능 버튼 영역 */}
      <div className="flex-shrink-0 pt-4 border-t border-blue-200">
          <div>
        <div className="space-y-4">
          <Button
            label="대화 기록 초기화"
            icon={<ArrowPathIcon className="h-5 w-5 mr-2" />}
            className="w-full pl-4 pr-4 h-10 rounded-full border border-gray-400 bg-white text-gray-700 flex items-center justify-center text-base font-semibold hover:bg-gray-100"
            onClick={onClearChatHistory}
          />
          {isLoggedIn && (
            <Button
              label="계정 삭제"
              icon={<TrashIcon className="h-5 w-5 mr-2" />}
              className="w-full pl-4 pr-4 h-10 rounded-full border border-red-500 bg-red-500 text-white flex items-center justify-center text-base font-semibold hover:bg-red-600"
              onClick={onDeleteAccount}
            />
          )}
        </div>
      </div>

      <div className="mt-auto">
        <div
          id={boardLinkId}
          className={`flex items-center py-2 px-3 rounded-md transition-colors duration-200 ${
            isLoggedIn ? 'hover:bg-blue-200 cursor-pointer' : 'opacity-50 cursor-not-allowed'
          }`}
          onClick={handleNavigateToBoard}
        >
          <DocumentTextIcon className="h-6 w-6 mr-2 text-gray-700" />
          <span className="font-medium text-gray-700">문의/건의 페이지로 이동하기</span>
        </div>
        
        {!isLoggedIn && (
          <Tooltip target={`#${boardLinkId}`} position="bottom">
            회원가입 후 이용 가능합니다.
          </Tooltip>
        )}
      </div>
      </div>
    </div>
  );
}