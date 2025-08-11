// src/app/(board)/new/page.tsx
import InquiryForm from '@/components/board/InquiryForm';
import React from 'react';

interface PageProps {
  searchParams: { [key: string]: string | string[] | undefined };
}

export default function NewInquiryPage({ searchParams }: PageProps) {
  const idParam = searchParams.id;

  // 만약 idParam이 배열이면 첫 번째 요소를, 아니면 그 값을 그대로 사용합니다.
  const inquiryId = Array.isArray(idParam) ? idParam[0] : idParam;

  return (
    <InquiryForm inquiryId={inquiryId} />
  );
}