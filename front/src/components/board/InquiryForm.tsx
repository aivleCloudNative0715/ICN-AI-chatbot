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
        // const response = await fetch(`/api/inquiries/${inquiryId}`, {
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
    <div className="flex-grow p-6 bg-blue-50">
      <div className="bg-white p-8 rounded-lg shadow-md max-w-2xl mx-auto">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">
          {isEditMode ? '문의/건의 수정' : '새 문의/건의 작성'}
        </h2>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label htmlFor="title" className="block text-gray-700 text-sm font-bold mb-2">
              <span className="text-red-500">*</span> 제목
            </label>
            <InputText
              id="title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="제목을 입력해주세요."
              className={`w-full p-2 border rounded-md ${errors.title ? 'p-invalid' : ''}`}
            />
            {errors.title && <small className="p-error text-red-500">{errors.title}</small>}
          </div>

          <div className="mb-4 relative">
            <label htmlFor="category" className="block text-gray-700 text-sm font-bold mb-2">
              <span className="text-red-500">*</span> 카테고리
            </label>
            <Dropdown
              id="category"
              value={category}
              options={categories}
              onChange={(e) => setCategory(e.value)}
              placeholder="카테고리를 선택해주세요."
              className={`w-full ${errors.category ? 'p-invalid' : ''}`}
              optionLabel="label"
              optionValue="value"
            />
            {errors.category && <small className="p-error text-red-500">{errors.category}</small>}
          </div>

          <div className="mb-6">
            <label htmlFor="content" className="block text-gray-700 text-sm font-bold mb-2">
              <span className="text-red-500">*</span> Contents
            </label>
            <InputTextarea
              id="content"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              rows={10}
              cols={30}
              placeholder="내용을 입력해주세요."
              className={`w-full p-2 border rounded-md ${errors.content ? 'p-invalid' : ''}`}
              autoResize
            />
            {errors.content && <small className="p-error text-red-500">{errors.content}</small>}
          </div>

          <div className="flex justify-center gap-4">
            <Button
              type="submit"
              label="등록"
              className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            />
            <Button
              type="button"
              label="취소"
              className="px-6 py-2 bg-gray-300 text-gray-800 rounded-md hover:bg-gray-400 transition-colors"
              onClick={handleCancel}
            />
          </div>
        </form>
      </div>
    </div>
  );
}