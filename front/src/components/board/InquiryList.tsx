// src/components/board/InquiryList.tsx
'use client';

import React, { useState, useEffect } from 'react';
import { Accordion, AccordionTab } from 'primereact/accordion';
import { InquiryDto } from '@/lib/types';

// // 1. 문의사항에 대한 구체적인 타입을 정의합니다. (any 대신)
// interface Inquiry {
//   id: number;
//   title: string;
//   content: string;
//   userId: string;
//   answer?: string;
// }

interface Props {
  inquiries: InquiryDto[];
  isLoading: boolean;
  error: string | null;
}

// // 임시 데이터: 실제로는 API를 통해 데이터를 가져와야 합니다.
// const allInquiries: Inquiry[] = [
//   { id: 1, userId: 'user123', title: '배송 관련 문의', content: '언제쯤 배송되나요?', answer: '내일 도착 예정입니다.' },
//   { id: 2, userId: 'user456', title: '상품 재고 문의', content: '이 상품 재입고 되나요?', answer: '죄송하지만 단종된 상품입니다.' },
//   { id: 3, userId: 'user123', title: '환불 규정 문의', content: '환불은 어떻게 진행되나요?', answer: '구매 후 7일 이내에 가능합니다.' },
//   { id: 4, userId: 'user789', title: '사이트 오류 문의', content: '로그인이 안돼요.' }, // 답변 없는 문의
// ];

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

export default function InquiryList({ inquiries, isLoading, error }: Props) {
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
          // inquiryId가 PK이므로 key로 사용
          <AccordionTab key={inquiry.inquiryId} header={inquiry.title}>
            {/* 상세 내용은 별도 페이지나 모달에서 보여주는 것이 일반적 */}
            <p className="m-0 text-board-dark">
              상태: {inquiry.status} | 긴급도: {inquiry.urgency}
            </p>
            {/* 여기에 수정/삭제 버튼 추가 가능 */}
          </AccordionTab>
        ))}
      </Accordion>
    </div>
  );
}