// src/app/(board)/new/page.tsx
import InquiryForm from '@/components/board/InquiryForm';
import React from 'react';

// ✅ 타입을 Promise로 직접 감싸서 정의합니다.
interface NewInquiryPageProps {
  params: Promise<{ [key: string]: string | string[] }>;
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}

// ✅ 컴포넌트는 async 함수로 유지하고, 내부에서 props를 await로 풀어줍니다.
export default async function NewInquiryPage({ searchParams }: NewInquiryPageProps) {
  // Promise로 감싸인 searchParams를 await로 기다려서 실제 객체를 얻습니다.
  const resolvedSearchParams = await searchParams;

  const inquiryIdToEdit = Array.isArray(resolvedSearchParams.id)
    ? resolvedSearchParams.id[0]
    : resolvedSearchParams.id;

  return (
    <InquiryForm inquiryId={inquiryIdToEdit} />
  );
}