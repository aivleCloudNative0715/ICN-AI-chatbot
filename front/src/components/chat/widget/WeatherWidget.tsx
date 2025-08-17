// src/components/chat/widget/WeatherWidget.tsx
'use client';

import React from 'react';
import { ArrivalWeatherInfo } from '@/lib/types';

interface WeatherWidgetProps {
  weather: ArrivalWeatherInfo | null;
  isLoading: boolean;
}

export default function WeatherWidget({ weather, isLoading }: WeatherWidgetProps) {
  if (isLoading) {
    return (
      <div className="bg-white p-3 rounded-lg shadow-sm border border-gray-200 text-center">
        <p className="text-gray-500">날씨 정보 로딩 중...</p>
      </div>
    );
  }

  if (!weather) {
    return null; // 날씨 정보가 없으면 위젯을 표시하지 않음
  }

  return (
    <div className="bg-white p-3 rounded-lg shadow-sm border border-gray-200">
      <h3 className="font-bold text-gray-800 mb-2 text-base">☀️ 현재 공항 날씨</h3>
      <div className="flex items-center justify-center space-x-4 mt-2">
        <img src={weather.wimage} alt="날씨 아이콘" className="w-12 h-12" />
        <p className="text-3xl font-bold text-gray-800">{weather.temp}°C</p>
      </div>
    </div>
  );
}