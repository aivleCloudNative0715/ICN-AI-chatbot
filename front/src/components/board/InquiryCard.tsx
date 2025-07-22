// src/components/board/InquiryCard.tsx
'use client';

import React, { useState } from 'react';
import { Button } from 'primereact/button';
import {
  TrashIcon,
  PencilIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  ChatBubbleBottomCenterTextIcon, // Q 아이콘
  ChatBubbleLeftRightIcon // A 아이콘
} from '@heroicons/react/24/outline';
import { InquiryListItem, InquiryAnswer } from '@/lib/types';
import { useRouter } from 'next/navigation';

interface InquiryCardProps {
  inquiry: InquiryListItem;
  isMyInquiries?: boolean; // '내 문의/건의 사항' 화면인지 여부
  // onEdit, onDelete 함수는 실제 API 호출을 담당할 부모 컴포넌트(InquiryList)에서 전달받습니다.
  onEdit?: (inquiryId: string) => void;
  onDelete?: (inquiryId: string) => void;
}

export default function InquiryCard({ inquiry, isMyInquiries = false, onEdit, onDelete }: InquiryCardProps) {
  const [isContentExpanded, setIsContentExpanded] = useState(false);
  const router = useRouter();

  const MAX_PREVIEW_LENGTH = 300; // UI/UX 문서에 따라 300자까지 미리보기

  const displayContent = isContentExpanded ? inquiry.content : inquiry.content.substring(0, MAX_PREVIEW_LENGTH);
  const needsTruncation = inquiry.content.length > MAX_PREVIEW_LENGTH;

  const handleEditClick = () => {
    if (onEdit) onEdit(inquiry.inquiry_id);
    else router.push(`/board/new?id=${inquiry.inquiry_id}`); // 수정 페이지로 이동
  };

  const handleDeleteClick = () => {
    if (confirm('정말 삭제하시겠습니까?')) {
      if (onDelete) onDelete(inquiry.inquiry_id);
      // 실제 삭제 API 호출 로직은 InquiryList에서 처리
      console.log(`Deleting inquiry: ${inquiry.inquiry_id}`);
    }
  };

  return (
    <div className="bg-white p-5 rounded-lg shadow-sm border border-gray-200 mb-4">
      {/* Question (문의/건의) 섹션 */}
      <div className="flex items-start mb-3 relative">
        <ChatBubbleBottomCenterTextIcon className="h-6 w-6 text-gray-600 mr-3 mt-1 flex-shrink-0" />
        <div className="flex-grow">
          <h3 className="font-semibold text-lg text-gray-900 mb-1">{inquiry.title}</h3>
          <p className="text-gray-700 text-sm whitespace-pre-wrap">
            {displayContent}
            {needsTruncation && !isContentExpanded && '...'}
          </p>
        </div>

        {/* 삭제, 수정 버튼 (로그인 사용자 및 내 문의/건의에서만) */}
        {isMyInquiries && (
          <div className="flex items-center absolute -top-2 right-0"> {/* UI/UX 이미지 위치에 맞춤 */}
            {/* 삭제 버튼 */}
            <Button
              icon={<TrashIcon className="h-4 w-4" />}
              className="p-button-text p-button-sm !text-gray-500 hover:!bg-gray-100 !p-1"
              onClick={handleDeleteClick}
              pt={{ root: { className: '!min-w-fit !h-auto !rounded-md' } }}
            />
            [cite_start]{/* 수정 버튼: '미처리' 상태이고 내 문의일 경우에만 활성화 [cite: 1] */}
            {inquiry.status === '미처리' && (
              <Button
                icon={<PencilIcon className="h-4 w-4" />}
                className="p-button-text p-button-sm !text-gray-500 hover:!bg-gray-100 !p-1"
                onClick={handleEditClick}
                pt={{ root: { className: '!min-w-fit !h-auto !rounded-md' } }}
              />
            )}
          </div>
        )}

        {/* 내용 펼치기/접기 버튼 (내용이 길 경우에만) */}
        {needsTruncation && (
          <Button
            icon={isContentExpanded ? <ChevronUpIcon className="h-5 w-5" /> : <ChevronDownIcon className="h-5 w-5" />}
            className="p-button-text p-button-sm !text-gray-600 hover:!bg-gray-100 !p-1 ml-2 self-start"
            onClick={() => setIsContentExpanded(!isContentExpanded)}
            pt={{ root: { className: '!min-w-fit !h-auto !rounded-md' } }}
          />
        )}
      </div>

      {/* Answer (답변) 섹션 (답변이 있는 경우에만 표시) */}
      {inquiry.hasAnswer && (
        <div className="border-t border-gray-200 pt-3 mt-3">
          <div className="flex items-start">
            <ChatBubbleLeftRightIcon className="h-6 w-6 text-gray-600 mr-3 mt-1 flex-shrink-0" />
            <div className="flex-grow">
              <h4 className="font-semibold text-base text-gray-800 mb-1">답변</h4>
              <p className="text-gray-700 text-sm whitespace-pre-wrap">
                {inquiry.answerContentPreview || '답변 내용입니다.'} {/* 실제 답변 내용으로 교체 필요 */}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}