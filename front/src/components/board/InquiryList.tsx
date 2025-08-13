// src/components/board/InquiryList.tsx
'use client';

import React, { useState, useEffect } from 'react';
import { Accordion, AccordionTab } from 'primereact/accordion';
import { InquiryDto } from '@/lib/types';
import { Button } from 'primereact/button'; 

interface Props {
  inquiries: InquiryDto[];
  isLoading: boolean;
  error: string | null;
  currentUserId: string;
  onDelete: (inquiryId: number) => void;
  onEdit: (inquiryId: number) => void;
}

// export default function InquiryList({ isMyInquiries, currentUserId }: Props) {
//   // 3. 컴포넌트 내부에서 보여줄 문의 목록 상태를 관리합니다.
//   const [inquiries, setInquiries] = useState<Inquiry[]>([]);

//   // 4. isMyInquiries 값에 따라 보여줄 데이터를 결정합니다.
//   useEffect(() => {
//     if (isMyInquiries) {
//       // "내 문의"가 참이면, 현재 사용자 ID와 일치하는 문의만 필터링합니다.
//       const myData = allInquiries.filter(inquiry => inquiry.userId === currentUserId);
//       setInquiries(myData);
//     } else {
//       // "내 문의"가 아니면 (예: 전체 문의 목록) 모든 데이터를 보여줍니다.
//       setInquiries(allInquiries);
//     }
//   }, [isMyInquiries, currentUserId]); // props가 변경될 때마다 이 로직이 다시 실행됩니다.

//   // 표시할 문의가 없을 때의 UI
//   if (inquiries.length === 0) {
//     return <div className="p-4 text-center text-board-dark">표시할 문의사항이 없습니다.</div>;
//   }

//   return (
//     <div className="card">
//       <Accordion multiple activeIndex={[0]}
//         pt={{
//           root: { className: 'bg-board-primary' },
//           accordiontab: {
//             headerAction: { className: 'flex items-center w-full text-board-dark border border-board-dark rounded-t-md bg-board-light' },
//             headerTitle: { className: 'font-bold' },
//             content: { className: 'border-b border-s border-e border-board-dark bg-board-primary p-4' }
//           }
//         }}
//       >
//         {inquiries.map((inquiry) => (
//           <AccordionTab key={inquiry.id} header={inquiry.title}>
//             <p className="m-0 text-board-dark">
//               {inquiry.content}
//             </p>
//             {inquiry.answer ? (
//               <>
//                 <p className="m-0 mt-3 font-bold text-board-dark">답변:</p>
//                 <p className="m-0 text-board-dark">{inquiry.answer}</p>
//               </>
//             ) : (
//               <p className="m-0 mt-3 text-red-500">아직 답변이 등록되지 않았습니다.</p>
//             )}
//           </AccordionTab>
//         ))}
//       </Accordion>
//     </div>
//   );
// }

export default function InquiryList({ inquiries, isLoading, error, currentUserId, onDelete, onEdit }: Props) {
  // 로딩 중일 때 UI
  if (isLoading) {
    return <div className="p-4 text-center text-board-dark">목록을 불러오는 중...</div>;
  }
  
  // 에러 발생 시 UI
  if (error) {
    return <div className="p-4 text-center text-red-500">{error}</div>;
  }

  // 데이터가 없을 때 UI
  if (inquiries.length === 0) {
    return <div className="p-4 text-center text-board-dark">표시할 내용이 없습니다.</div>;
  }

  return (
    <div className="card">
      <Accordion multiple>
        {inquiries.map((inquiry) => (
          <AccordionTab key={inquiry.inquiryId} header={inquiry.title}>
            <div className="text-board-dark">
              <p>상태: {inquiry.status} | 긴급도: {inquiry.urgency}</p>
              {/* 현재 로그인한 사용자의 글일 경우에만 수정/삭제 버튼 표시 */}
              {inquiry.userId === currentUserId && (
                <div className="mt-4 flex gap-2">
                  <Button
                    label="수정"
                    icon="pi pi-pencil"
                    className="p-button-sm p-button-secondary"
                    onClick={() => onEdit(inquiry.inquiryId)}
                  />
                  <Button
                    label="삭제"
                    icon="pi pi-trash"
                    className="p-button-sm p-button-danger"
                    onClick={() => onDelete(inquiry.inquiryId)}
                  />
                </div>
              )}
            </div>
          </AccordionTab>
        ))}
      </Accordion>
    </div>
  );
}