// src/components/board/InquiryList.tsx
'use client';

import React, { useState } from 'react';
import { Accordion, AccordionTab } from 'primereact/accordion';

interface Props {
  inquiries: any
}

export default function InquiryList({inquiries}: Props) {
//   // 더미 데이터 (API 연결 전 퍼블리싱용)
//   const inquiries = [
//     {
//       id: 1,
//       type: 'Q',
//       title: '질문 제목입니다.',
//       content: '질문 내용입니다.',
//       answer: '답변 내용입니다.',
//     },
//     {
//       id: 2,
//       type: 'Q',
//       title: '질문 제목입니다.',
//       content: '질문 내용입니다. 최대 300자까지 보여주고, 더보기로 내용을 확인할 수 있도록 ....',
//       answer: '', // 답변이 없는 경우
//     },
//     {
//       id: 3,
//       type: 'Q',
//       title: '질문 제목입니다.',
//       content: '질문 내용입니다. 최대 300자까지 보여주고, 더보기로 내용을 확인할 수 있도록 ....',
//       answer: '', // 답변이 없는 경우
//     },
//   ];

  return (
    <div className="card">
      <Accordion multiple activeIndex={[0]}>
        {inquiries.map((inquiry: any) => (
          <AccordionTab key={inquiry.id} header={inquiry.title}>
            <p className="m-0">
              {inquiry.content}
            </p>
            {inquiry.answer && (
              <>
                <p className="m-0 mt-3 font-bold">답변:</p>
                <p className="m-0">{inquiry.answer}</p>
              </>
            )}
            {!inquiry.answer && (
              <p className="m-0 mt-3 text-red-500">아직 답변이 등록되지 않았습니다.</p>
            )}
          </AccordionTab>
        ))}
      </Accordion>
    </div>
  );
}