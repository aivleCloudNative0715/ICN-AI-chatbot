// src/components/board/InquiryForm.tsx
'use client';

import React, { useState, useEffect } from 'react';
import { Dropdown } from 'primereact/dropdown';
import { useRouter } from 'next/navigation';
import { createInquiry, getInquiryDetail, updateInquiry } from '@/lib/api';
import { useAuth } from '@/contexts/AuthContext';

interface InquiryFormProps {
  inquiryId?: string;
}

export default function InquiryForm({ inquiryId }: InquiryFormProps) {
  const router = useRouter();
  const { user, token } = useAuth();
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  // 카테고리 타입을 백엔드 Enum과 맞춥니다.
  const [category, setCategory] = useState<'INQUIRY' | 'SUGGESTION' | null>(null);
  const [errors, setErrors] = useState<{ title?: string; content?: string; category?: string }>({});
  const [isSubmitting, setIsSubmitting] = useState(false); // 제출 중 상태
  const isEditMode = !!inquiryId;

  // 카테고리 목록을 백엔드 Enum에 맞게 수정
  const categories = [
    { label: '문의', value: 'INQUIRY' },
    { label: '건의', value: 'SUGGESTION' },
  ];

  // 1. 수정 모드일 때, useEffect를 사용해 기존 데이터를 불러옵니다.
  useEffect(() => {
    // inquiryId가 있고, isEditMode가 true일 때만 실행
     if (isEditMode && inquiryId && user && token) {
      const fetchInquiryData = async () => {
        try {
          const inquiryData = await getInquiryDetail(Number(inquiryId), token);
          setTitle(inquiryData.title);
          setContent(inquiryData.content);
          setCategory(inquiryData.category);
        } catch (err) {
          alert(err instanceof Error ? err.message : '데이터를 불러올 수 없습니다.');
          router.back(); // 오류 발생 시 이전 페이지로 이동
        }
      };
      fetchInquiryData();
    }
  }, [inquiryId, isEditMode, router, user, token]); // 의존성 배열에 router, userId 추가

  const validate = () => {
    const newErrors: { title?: string; content?: string; category?: string } = {};
    if (!title.trim()) newErrors.title = '제목을 입력해주세요.';
    if (!content.trim()) newErrors.content = '내용을 입력해주세요.';
    if (!category) newErrors.category = '카테고리를 선택해주세요.';
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!user || !token) {
        alert('로그인이 필요합니다.');
        return;
    }

    if (!validate() || isSubmitting) return;

    setIsSubmitting(true);
    try {
      const dataToSend = { title, content, category: category! };
      if (isEditMode && inquiryId) {
        await updateInquiry(Number(inquiryId), dataToSend, token);
        alert('문의/건의가 성공적으로 수정되었습니다.');
      } else {
        await createInquiry(dataToSend, token);
        alert('새 문의/건의가 성공적으로 작성되었습니다.');
      }
      router.push('/board');
    } catch (err) {
      alert(`오류 발생: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancel = () => {
    router.back(); // 이전 페이지로 돌아가기
  };

  return (
    <div className="flex-grow p-8 bg-board-primary min-h-screen">
      <div className="bg-white p-10 rounded-md shadow-md max-w-xl mx-auto">
        <h2 className="text-xl font-semibold text-board-dark mb-6 text-center">
          {isEditMode ? '문의/건의 수정' : '새 문의/건의 작성'}
        </h2>
        <form onSubmit={handleSubmit} className="space-y-6">
          
          {/* 제목 입력 */}
          <div>
            <label htmlFor="title" className="block text-sm font-medium text-board-dark mb-2">
              <span className="text-red-500">*</span> 제목을 입력해주세요.
            </label>
            <input
              id="title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="제목을 입력해주세요."
              className={`w-full px-2 py-2 text-gray-700 border-b-2 border-gray-300 bg-transparent focus:outline-none focus:border-board-dark transition-all ${
                errors.title ? 'border-red-500' : ''
              }`}
            />
            {errors.title && <p className="text-red-500 text-sm mt-1">{errors.title}</p>}
          </div>

          {/* 카테고리 */}
          <div className="flex justify-end">
            <div className="w-1/2">
              <label htmlFor="category" className="block text-sm font-medium text-board-dark mb-2">
                <span className="text-red-500">*</span> 카테고리를 선택해주세요.
              </label>
              <Dropdown
                id="category"
                value={category}
                options={categories}
                onChange={(e) => setCategory(e.value)}
                placeholder="카테고리를 선택해주세요."
                className={`w-full border border-board-dark bg-transparent focus:border-board-dark ${
                  errors.category ? 'border-red-500' : ''
                }`}
                optionLabel="label"
                optionValue="value"
              />
              {errors.category && <p className="text-red-500 text-sm mt-1">{errors.category}</p>}
            </div>
          </div>

          {/* 내용 입력 */}
          <div>
            <label htmlFor="content" className="block text-sm font-medium text-board-dark mb-2">
              <span className="text-red-500">*</span> Contents
            </label>
            <textarea
              id="content"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              rows={8}
              placeholder="내용을 입력해주세요."
              className={`w-full px-2 py-2 border-2 border-gray-300 rounded-md bg-transparent focus:outline-none focus:border-board-dark transition-all ${
                errors.content ? 'border-red-500' : ''
              }`}
            />
            {errors.content && <p className="text-red-500 text-sm mt-1">{errors.content}</p>}
          </div>

          {/* 버튼 */}
          <div className="flex justify-center gap-4">
            <button
              type="submit"
              disabled={isSubmitting} 
              className="px-8 py-2 bg-board-dark text-white rounded hover:bg-board-light transition-colors"
            >
              {isSubmitting ? '등록 중...' : '등록'}
            </button>
            <button
              type="button"
              onClick={handleCancel}
              className="px-8 py-2 border border-board-dark text-board-dark rounded hover:bg-board-light transition-colors"
            >
              취소
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}