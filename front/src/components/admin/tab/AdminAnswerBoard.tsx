// AdminAnswerBoard.tsx
'use client';

import React, { useState } from 'react';
import { Button } from 'primereact/button';
import { InputTextarea } from 'primereact/inputtextarea';
import { Dropdown } from 'primereact/dropdown';
import { UserIcon, CalendarDaysIcon, TagIcon } from '@heroicons/react/24/outline';
import CustomPriorityDropdown from '@/components/CustomPriorityDropdown'; // Assuming you have this component

interface AdminAnswerBoardProps {
  inquiry: {
    id: number;
    title: string;
    content: string;
    author: string;
    date: string;
    category: string;
    priority: string;
    status: string;
  };
  onBack: () => void;
  onRegister: (answerContent: string, newPriority: string) => void;
}

export default function AdminAnswerBoard({ inquiry, onBack, onRegister }: AdminAnswerBoardProps) {
  const [answerContent, setAnswerContent] = useState('');
  const [currentPriority, setCurrentPriority] = useState(inquiry.priority);

  const handleRegister = () => {
    onRegister(answerContent, currentPriority);
  };

  return (
    <div className="p-4 bg-white rounded-lg shadow-md">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">관리자 대시보드</h1>
        {/* 로그아웃 버튼은 AdminHeader에서 관리되므로 여기서는 제거 */}
      </div>

      {/* Inquiry Details Section */}
      <div className="border border-[#C5C5C5] rounded-lg p-6 mb-6">
        <h2 className="text-xl font-bold mb-4">{inquiry.title}</h2>
        <div className="flex items-center gap-4 text-sm text-gray-500 mb-4">
          <div className='flex gap-2'> <UserIcon className="h-4 w-4" /> {inquiry.author}</div>
          <div className='flex gap-2'> <CalendarDaysIcon className="h-4 w-4" /> {inquiry.date}</div>
          <div className='flex gap-2'> <TagIcon className="h-4 w-4" /> {inquiry.category}</div>
        </div>
        <p className="text-gray-700 leading-relaxed mb-4">{inquiry.content}</p>
        <div className="flex justify-end">
          <CustomPriorityDropdown
            value={currentPriority}
            onChange={(newValue) => setCurrentPriority(newValue)}
          />
        </div>
      </div>

      {/* Answer Section */}
      <div className="mb-6">
        <label htmlFor="answer" className="flex block text-lg font-bold text-gray-800 mb-2">
          <p className='text-red-500 mr-2'>*</p> 답변
        </label>
        <InputTextarea
          id="answer"
          value={answerContent}
          onChange={(e) => setAnswerContent(e.target.value)}
          rows={10}
          className="w-full border border-gray-300 rounded-md p-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          placeholder="관리자 답변 작성 텍스트 박스입니다."
        />
      </div>

      {/* Action Buttons */}
      <div className="flex justify-center gap-4">
        <Button label="등록" onClick={handleRegister} className="px-3 py-1.5 text-sm bg-white border border-black rounded-md hover:bg-black hover:text-white transition duration-300" />
        <Button label="취소" onClick={onBack} className="px-3 py-1.5 text-sm bg-white border border-black rounded-md hover:bg-black hover:text-white transition duration-300" />
      </div>
    </div>
  );
}