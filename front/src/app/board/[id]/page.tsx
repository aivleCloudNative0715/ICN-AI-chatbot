'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { getInquiryDetail, deleteInquiry } from '@/lib/api';
import { InquiryDetailDto } from '@/lib/types';
import { Button } from 'primereact/button';

export default function InquiryDetailPage() {
  const router = useRouter();
  const params = useParams();
  const { user, token } = useAuth();
  
  const [inquiry, setInquiry] = useState<InquiryDetailDto | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const inquiryId = Number(params.id);

  useEffect(() => {
    if (!token) {
      setIsLoading(false);
      setError('이 페이지에 접근하려면 로그인이 필요합니다.');
      return;
    }
    if (inquiryId) {
      getInquiryDetail(inquiryId, token)
        .then(data => setInquiry(data))
        .catch(err => setError('문의/건의 내용을 불러오는데 실패했습니다.'))
        .finally(() => setIsLoading(false));
    }
  }, [inquiryId, token]);

  const handleDelete = async () => {
    if (!token || !inquiry) return;
    if (window.confirm('정말로 이 문의/건의를 삭제하시겠습니까?')) {
      try {
        await deleteInquiry(inquiry.inquiryId, token);
        alert('문의/건의가 삭제되었습니다.');
        router.push('/board');
      } catch (err) {
        alert(err instanceof Error ? err.message : '삭제 중 오류 발생');
      }
    }
  };

  if (isLoading) return <div className="p-4 text-center">로딩 중...</div>;
  if (error) return <div className="p-4 text-center text-red-500">{error}</div>;
  if (!inquiry) return <div className="p-4 text-center">문의/건의 내용을 찾을 수 없습니다.</div>;

  const isAuthor = user?.userId === inquiry.userId;

  return (
    <div className="bg-white p-8 rounded-md shadow-lg max-w-4xl mx-auto my-8">
      {/* Header Section */}
      <div className="border-b pb-4 mb-6">
        <h1 className="text-3xl font-bold text-gray-800">{inquiry.title}</h1>
        <div className="flex justify-between items-center text-sm text-gray-500 mt-3">
          <div>
            <span>카테고리: <strong>{inquiry.category === 'INQUIRY' ? '문의' : '건의'}</strong></span> | 
            <span className="ml-2">상태: <strong>{inquiry.status === 'PENDING' ? '답변 대기' : '답변 완료'}</strong></span>
          </div>
          <div>
            <span>작성일: {new Date(inquiry.createdAt).toLocaleDateString()}</span>
          </div>
        </div>
      </div>
      
      {/* Content Section */}
      <div className="prose max-w-none mb-8 min-h-[150px]">
        <p className="text-gray-700 whitespace-pre-wrap">{inquiry.content}</p>
      </div>
      
      {/* 관리자 답변 표시 */}
      <div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
        <h3 className="text-xl font-semibold text-gray-800 mb-3">관리자 답변</h3>
        {inquiry.answer ? (
          <p className="text-gray-600 whitespace-pre-wrap">{inquiry.answer}</p>
        ) : (
          <p className="text-gray-500 italic">아직 등록된 답변이 없습니다.</p>
        )}
      </div>

      {/* Button Section */}
      <div className="mt-10 flex justify-between items-center">
        {/* 목록으로 버튼 (항상 보임) */}
        <Button 
          label="목록으로" 
          icon="pi pi-bars" 
          onClick={() => router.push('/board')}
          className="p-button-text"
        />
        
        {/* 내가 쓴 글일 때만 보이는 버튼 그룹 */}
        {isAuthor && (
          <div className="flex gap-4">
            {/* 답변이 없을 때만 '수정' 버튼이 보임 */}
            {!inquiry.answer && (
              <Button 
                label="수정" 
                icon="pi pi-pencil" 
                onClick={() => router.push(`/board/new?id=${inquiry.inquiryId}`)}
                className="p-button-secondary"
              />
            )}
            {/* 삭제 버튼은 항상 보임 */}
            <Button 
              label="삭제" 
              icon="pi pi-trash" 
              onClick={handleDelete}
              className="p-button-danger"
            />
          </div>
        )}
      </div>
    </div>
  );
}