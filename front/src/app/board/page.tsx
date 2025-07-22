// src/app/(board)/page.tsx
import InquiryList from '@/components/board/InquiryList';
import React from 'react';

export default function BoardPage() {
  // 전체 문의/건의 사항은 category prop을 전달하지 않거나, URL 쿼리 파라미터로 처리할 수 있습니다.
  // 이 예시에서는 쿼리 파라미터를 사용하여 '건의 사항'도 필터링할 수 있도록 합니다.
  // useSearchParams를 사용해야 하지만, 서버 컴포넌트이므로 일단은 category prop을 사용하지 않습니다.
  // 클라이언트 컴포넌트인 InquiryList에서 useSearchParams를 직접 사용하도록 합니다.
  return (
    <InquiryList />
  );
}