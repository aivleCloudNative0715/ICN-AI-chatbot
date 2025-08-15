// src/components/admin/tab/AdminAnswerBoard.tsx
'use client';

import React, { useState, useEffect } from 'react';
import { Button } from 'primereact/button';
import { InputTextarea } from 'primereact/inputtextarea';
import { UserIcon, CalendarDaysIcon, TagIcon } from '@heroicons/react/24/outline';
import CustomPriorityDropdown from '@/components/CustomPriorityDropdown';
import { useMutation, useQueryClient, useQuery } from '@tanstack/react-query';
import { processAdminAnswer, getAdminInquiryDetail, updateInquiryUrgency } from '@/lib/api';
import { AdminInquiryDetailDto, Urgency } from '@/lib/types';
import { useAuth } from '@/contexts/AuthContext';

interface AdminAnswerBoardProps {
  inquiry: { inquiryId: number };
  onBack: () => void;
}

export default function AdminAnswerBoard({ inquiry, onBack }: AdminAnswerBoardProps) {
  const { token, user } = useAuth();
  const queryClient = useQueryClient();
  const [answerContent, setAnswerContent] = useState('');

  // --- 상세 데이터를 불러오는 useQuery ---
  const { data: inquiryDetail, isLoading, isError, error } = useQuery({
    // onSuccess 옵션을 제거합니다.
    queryKey: ['adminInquiryDetail', inquiry.inquiryId],
    queryFn: () => {
      if (!token) throw new Error("인증되지 않았습니다.");
      return getAdminInquiryDetail(inquiry.inquiryId, token);
    },
    enabled: !!token && !!inquiry.inquiryId,
  });

  // --- useEffect를 사용하여 onSuccess 로직을 처리 ---
  useEffect(() => {
    // inquiryDetail 데이터가 성공적으로 로드되면 answerContent 상태를 업데이트합니다.
    if (inquiryDetail?.answer) {
      setAnswerContent(inquiryDetail.answer);
    }
  }, [inquiryDetail]);

  // --- 답변 등록을 위한 useMutation ---
  const processAnswerMutation = useMutation({
    mutationFn: (newAnswer: string) => {
      // user와 user.adminId의 존재를 명확히 확인합니다.
      if (!token || !user?.adminId) {
        throw new Error("유효한 관리자 정보가 없습니다.");
      }
      return processAdminAnswer(inquiry.inquiryId, token, {
        adminId: user.adminId, // 이제 타입 에러가 발생하지 않습니다.
        content: newAnswer,
      });
    },
    onSuccess: () => {
      alert("답변이 성공적으로 등록되었습니다.");
      queryClient.invalidateQueries({ queryKey: ['adminInquiries'] });
      queryClient.invalidateQueries({ queryKey: ['adminInquiryDetail', inquiry.inquiryId] });
      onBack();
    },
    onError: (error) => {
      alert(`답변 등록 실패: ${error.message}`);
    }
  });

  // 긴급도 수정 Mutation (AdminContentBoard와 동일)
  const updateUrgencyMutation = useMutation({
    mutationFn: ({ inquiryId, newPriority }: { inquiryId: number, newPriority: Urgency }) => {
              if (!token) throw new Error("인증되지 않았습니다.");
              return updateInquiryUrgency(inquiryId, token, newPriority);
          },
          onSuccess: () => {
              // 성공 시, 문의 목록 쿼리를 무효화하여 최신 데이터로 갱신
              queryClient.invalidateQueries({ queryKey: ['adminInquiries'] });
          },
          onError: (error) => {
              alert(`긴급도 수정 실패: ${error.message}`);
          }
  });

  const handleRegister = () => {
    if (!answerContent.trim()) {
      alert("답변 내용을 입력해주세요.");
      return;
    }
    processAnswerMutation.mutate(answerContent);
  };

  const handlePriorityChange = (id: number, newPriority: string) => {
    updateUrgencyMutation.mutate({ inquiryId: id, newPriority: newPriority as Urgency });
  };
  
  if (isLoading) return <div>상세 정보 로딩 중...</div>;
  if (isError || !inquiryDetail) return <div>오류가 발생했습니다: {error?.message || '데이터를 찾을 수 없습니다.'}</div>;
  
  
  return (
    <div className="p-4 bg-white rounded-lg shadow-md">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">관리자 대시보드</h1>
        {/* 로그아웃 버튼은 AdminHeader에서 관리되므로 여기서는 제거 */}
      </div>

      {/* 문의 상세 정보 섹션 (inquiryDetail 사용) */}
      <div className="border border-[#C5C5C5] rounded-lg p-6 mb-6">
        <h2 className="text-xl font-bold mb-4">{inquiryDetail.title}</h2>
        <div className="flex items-center gap-4 text-sm text-gray-500 mb-4 flex-wrap">
          <div className='flex gap-2 items-center'> <UserIcon className="h-4 w-4" /> {inquiryDetail.userId}</div>
          <div className='flex gap-2 items-center'> <CalendarDaysIcon className="h-4 w-4" /> {new Date(inquiryDetail.createdAt).toLocaleString()}</div>
          <div className='flex gap-2 items-center'> <TagIcon className="h-4 w-4" /> {inquiryDetail.category}</div>
        </div>
        <p className="text-gray-700 leading-relaxed mb-4 p-4 bg-gray-50 rounded min-h-[100px]">{inquiryDetail.content}</p>
        <div className="flex justify-end">
          <CustomPriorityDropdown
            value={inquiryDetail.urgency}
            onChange={(newValue) => handlePriorityChange(inquiryDetail.inquiryId, newValue)}
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

      {/* 버튼 부분 */}
      <div className="flex justify-center gap-4">
        <Button 
            label="등록" 
            onClick={handleRegister} 
            disabled={processAnswerMutation.isPending} // 로딩 중 비활성화
            className="..." 
        />
        <Button label="목록으로" onClick={onBack} className="..." />
      </div>
    </div>
  );
}