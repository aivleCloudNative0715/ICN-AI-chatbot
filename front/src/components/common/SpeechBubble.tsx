// src/components/common/SpeechBubble.tsx
'use client';

import React from 'react'; // useRef, useState, useEffect는 이제 필요 없음
import styles from './SpeechBubble.module.css';

interface SpeechBubbleProps {
  message: string;
  position?: 'left' | 'right';
  minWidth?: string;
  maxWidth?: string;
}

export default function SpeechBubble({
  message,
  position = 'right',
  minWidth = '50px', // 최소 너비
  maxWidth = 'fit-content', // 콘텐츠에 맞춤
}: SpeechBubbleProps) {

  const bubbleClasses = `${styles.bubble} px-3 py-1.5 text-xs text-gray-800`; // Tailwind 클래스 유지

  const tailPositionClass = position === 'left' ? styles.bubbleTailLeft : styles.bubbleTailRight;

  return (
    <div
      className={bubbleClasses.trim()}
      style={{
        minWidth: minWidth,
        maxWidth: maxWidth,
      }}
    >
      {message}
      <div className={tailPositionClass}></div>
    </div>
  );
}