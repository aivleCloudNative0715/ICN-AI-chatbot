// src/app/(board)/my-inquiries/page.tsx
import InquiryList from '@/components/board/InquiryList';
import React from 'react';

export default function MyInquiriesPage() {
  // TODO: 실제 로그인한 사용자의 ID를 Context 또는 서버 사이드에서 가져와야 합니다.
  const currentUserId = 'user123'; // 임시 사용자 ID

  return (
    <InquiryList isMyInquiries={true} currentUserId={currentUserId} />
  );
}