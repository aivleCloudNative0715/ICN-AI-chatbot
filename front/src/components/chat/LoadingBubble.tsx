// src/components/chat/LoadingBubble.tsx
'use client';

import React from 'react';
import { ProgressSpinner } from 'primereact/progressspinner';

export default function LoadingBubble() {
  return (
    <div className="flex justify-start mb-4">
      <div
        className="flex items-center space-x-3 max-w-lg p-3 rounded-xl bg-blue-50 text-gray-800 rounded-bl-none"
      >
        <ProgressSpinner 
          style={{width: '24px', height: '24px'}} 
          strokeWidth="6" 
          animationDuration=".5s"
        />
        <p className="text-sm sm:text-base italic">답변 생성중...</p>
      </div>
    </div>
  );
}