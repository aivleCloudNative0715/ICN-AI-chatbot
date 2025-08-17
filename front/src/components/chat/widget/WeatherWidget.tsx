// src/components/chat/widget/WeatherWidget.tsx
'use client';

import React, { useState, useEffect } from 'react';
import { getArrivalsWeather } from '@/lib/api';
import { ArrivalWeatherInfo } from '@/lib/types';

export default function WeatherWidget() {
  const [weather, setWeather] = useState<ArrivalWeatherInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchWeatherData = async () => {
      try {
        setIsLoading(true);
        const data = await getArrivalsWeather();
        // API가 도착편 목록을 반환하므로, 첫 번째 항목을 대표 날씨로 사용합니다.
        if (data && data.length > 0) {
          setWeather(data[0]);
        } else {
          setWeather(null); // 데이터가 없으면 날씨 정보 없음
        }
      } catch (error) {
        console.error("Failed to fetch weather data:", error);
        setWeather(null);
      } finally {
        setIsLoading(false);
      }
    };

    fetchWeatherData();
  }, []);

  if (isLoading) {
    return (
      <div className="bg-white p-3 rounded-lg shadow-sm border border-gray-200 text-center">
        <p className="text-gray-500">날씨 정보 로딩 중...</p>
      </div>
    );
  }

  // 날씨 정보가 없으면 위젯을 아예 표시하지 않음
  if (!weather) {
    return null;
  }

  return (
    <div className="bg-white p-3 rounded-lg shadow-sm border border-gray-200">
      <h3 className="font-bold text-gray-800 mb-2 text-base">☀️ 현재 공항 날씨</h3>
      <div className="flex items-center justify-center space-x-4 mt-2">
        {/* API가 제공하는 날씨 아이콘 이미지를 사용합니다. */}
        <img src={weather.wimage} alt="날씨 아이콘" className="w-12 h-12" />
        <p className="text-3xl font-bold text-gray-800">{weather.temp}°C</p>
      </div>
    </div>
  );
}