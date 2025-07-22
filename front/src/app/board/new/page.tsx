// src/app/(board)/new/page.tsx
import InquiryForm from '@/components/board/InquiryForm';
import React from 'react';

interface NewInquiryPageProps {
  searchParams?: {
    id?: string; // 수정 모드일 경우 문의 ID를 쿼리 파라미터로 받음
  };
}

export default function NewInquiryPage({ searchParams }: NewInquiryPageProps) {
  const inquiryIdToEdit = searchParams?.id;

  return (
    <InquiryForm inquiryId={inquiryIdToEdit} />
  );
}