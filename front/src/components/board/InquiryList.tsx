// src/components/board/InquiryList.tsx
'use client';

import React from 'react';
import { Accordion, AccordionTab } from 'primereact/accordion';

interface Props {
  inquiries: any
}

export default function InquiryList({inquiries}: Props) {
  return (
    <div className="card">
      <Accordion multiple activeIndex={[0]}
        pt={{
          root: { className: 'bg-board-primary' }, // Uses to pass attributes to the root's DOM element. Apply background to the accordion root
          accordiontab: { // Uses to pass attributes to accordion tabs.
            headerAction: { className: 'flex items-center w-full text-board-dark border border-board-dark rounded-t-md bg-board-light' }, // Uses to pass attributes to the headeraction's DOM element. Header styles from BoardSidebar
            headerTitle: { className: 'font-bold' }, // Uses to pass attributes to the headertitle's DOM element. Make header title bold
            content: { className: 'border-b border-s border-e border-board-dark bg-board-primary p-4' } // Uses to pass attributes to the content's DOM element. Content area background and padding
          }
        }}
      >
        {inquiries.map((inquiry: any) => (
          <AccordionTab key={inquiry.id} header={inquiry.title}>
            <p className="m-0 text-board-dark"> {/* Apply text color */}
              {inquiry.content}
            </p>
            {inquiry.answer && (
              <>
                <p className="m-0 mt-3 font-bold text-board-dark">답변:</p> {/* Apply text color */}
                <p className="m-0 text-board-dark">{inquiry.answer}</p> {/* Apply text color */}
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