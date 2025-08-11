// src/components/board/InquiryForm.tsx
'use client';

import React, { useState, useEffect } from 'react';
import { InputText } from 'primereact/inputtext';
import { InputTextarea } from 'primereact/inputtextarea';
import { Dropdown } from 'primereact/dropdown';
import { Button } from 'primereact/button';
import { useRouter } from 'next/navigation';
import { Inquiry } from '@/lib/types';

interface InquiryFormProps {
  inquiryId?: string; // 수정 모드일 경우 문의 ID
  initialData?: Inquiry; // 수정 모드일 경우 초기 데이터
}

export default function InquiryForm({ inquiryId, initialData }: InquiryFormProps) {
  const router = useRouter();
  const [title, setTitle] = useState(initialData?.title || '');
  const [content, setContent] = useState(initialData?.content || '');
  const [category, setCategory] = useState<'문의' | '건의' | null>(initialData?.category || null);
  const [errors, setErrors] = useState<{ title?: string; content?: string; category?: string }>({});
  const isEditMode = !!inquiryId;

  const categories = [
    { label: '문의', value: '문의' },
    { label: '건의', value: '건의' },
  ];

  useEffect(() => {
    if (isEditMode && !initialData) {
      // TODO: 수정 모드이고 initialData가 없는 경우, API-13-27039 (단일 문의/건의 내역 가져오기) 호출하여 데이터 로드
      // 현재는 예시 데이터를 사용
      const fetchInquiryData = async () => {
        // 실제 API 호출 시뮬레이션
        const mockInquiry: Inquiry = {
          inquiry_id: inquiryId!, user_id: 'user123', title: '로드된 문의 제목', content: '이것은 로드된 문의 내용입니다. 수정할 수 있습니다.', category: '문의', urgency: '보통', status: '미처리', created_at: '2025-07-20T10:00:00Z', updated_at: '2025-07-20T10:00:00Z', is_deleted: false
        };
        setTitle(mockInquiry.title);
        setContent(mockInquiry.content);
        setCategory(mockInquiry.category);
      };
      fetchInquiryData();
    }
  }, [inquiryId, isEditMode, initialData]);

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
    if (!validate()) {
      return;
    }

    const dataToSend = {
      title,
      content,
      category,
      // urgency는 기본값('보통')으로 설정 (API-13-27040)
      // status는 '미처리'로 초기 설정 (API-13-27040)
      // user_id는 백엔드에서 JWT 토큰으로 추출 (API-13-27040)
    };

    try {
      if (isEditMode) {
        // TODO: API-13-27041 (문의/건의 수정) 호출 로직
        console.log('문의/건의 수정 시도:', inquiryId, dataToSend);
        // const response = await fetch(`${API_BASE_URL}/api/inquiries/${inquiryId}`, {
        //   method: 'PUT',
        //   headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer YOUR_JWT_TOKEN` },
        //   body: JSON.stringify(dataToSend),
        // });
        // if (!response.ok) throw new Error('수정 실패');
        alert('문의/건의가 성공적으로 수정되었습니다.');
      } else {
        // TODO: API-13-27040 (문의/건의 작성) 호출 로직
        console.log('새 문의/건의 작성 시도:', dataToSend);
        // const response = await fetch('/api/inquiries', {
        //   method: 'POST',
        //   headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer YOUR_JWT_TOKEN` },
        //   body: JSON.stringify(dataToSend),
        // });
        // if (!response.ok) throw new Error('작성 실패');
        alert('새 문의/건의가 성공적으로 작성되었습니다.');
      }
      router.push('/board/my-inquiries'); // 작성/수정 후 내 문의/건의 목록으로 이동
    } catch (err) {
      alert(`오류 발생: ${err instanceof Error ? err.message : String(err)}`);
      console.error(err);
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
              className="px-8 py-2 bg-board-dark text-white rounded hover:bg-board-light transition-colors"
            >
              등록
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