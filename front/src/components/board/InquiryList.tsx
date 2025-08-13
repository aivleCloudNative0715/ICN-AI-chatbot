// src/components/board/InquiryList.tsx
'use client';

import React from 'react';
import { InquiryDto } from '@/lib/types';
import { useRouter } from 'next/navigation';
import { DataTable } from 'primereact/datatable'; // DataTable 임포트
import { Column } from 'primereact/column';     // Column 임포트

interface Props {
  inquiries: InquiryDto[];
  isLoading: boolean;
  error: string | null;
  first: number;
  rows: number;
  totalRecords: number;
  onPageChange: (event: any) => void;
}

export default function InquiryList({ inquiries, isLoading, error, first, rows, totalRecords, onPageChange }: Props) {
  const router = useRouter();

  // 날짜 포맷팅을 위한 템플릿
  const dateBodyTemplate = (rowData: InquiryDto) => {
    return new Date(rowData.createdAt).toLocaleDateString();
  };
  
  // 상태 표시를 위한 템플릿
  const statusBodyTemplate = (rowData: InquiryDto) => {
    const isPending = rowData.status === 'PENDING';
    const bgColor = isPending ? 'bg-yellow-100' : 'bg-green-100';
    const textColor = isPending ? 'text-yellow-800' : 'text-green-800';
    const text = isPending ? '답변 대기' : '답변 완료';
    return (
      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${bgColor} ${textColor}`}>
        {text}
      </span>
    );
  };

  return (
    <div className="card">
      <DataTable
        value={inquiries}
        loading={isLoading}
        // 페이지네이션 설정
        paginator
        lazy // 서버사이드 페이지네이션을 위해 lazy 모드 활성화
        first={first}
        rows={rows}
        totalRecords={totalRecords}
        onPage={onPageChange}
        rowsPerPageOptions={[5, 10, 20]}
        // 행 클릭 이벤트
        selectionMode="single"
        onRowSelect={(e) => router.push(`/board/${e.data.inquiryId}`)}
        // 스타일
        dataKey="inquiryId"
        emptyMessage={error || "표시할 내용이 없습니다."}
      >
        <Column field="inquiryId" header="번호" style={{ width: '10%' }} />
        {/* <Column field="category" header="카테고리" style={{ width: '15%' }} /> */}
        <Column field="title" header="제목" />
        <Column field="userId" header="작성자" style={{ width: '20%' }} />
        <Column field="status" header="상태" body={statusBodyTemplate} style={{ width: '15%' }} />
        <Column field="createdAt" header="작성일" body={dateBodyTemplate} style={{ width: '15%' }} />
      </DataTable>
    </div>
  );
}