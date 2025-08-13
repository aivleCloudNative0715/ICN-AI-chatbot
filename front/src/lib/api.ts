// src/lib/api.ts
import { InquiryDto, InquiryDetailDto, Page } from './types';

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;

const getAuthHeaders = (token: string | null) => {
  return {
    'Content-Type': 'application/json',
    ...(token && { 'Authorization': `Bearer ${token}` }),
  };
};

/** 전체 게시글 목록 조회 (GET /api/board) */
export const getAllInquiries = async (page: number = 0, size: number = 10): Promise<Page<InquiryDto>> => {
  const response = await fetch(`${API_BASE_URL}/api/board?page=${page}&size=${size}`);
  if (!response.ok) throw new Error('게시글 목록을 불러오는데 실패했습니다.');
  return response.json();
};

/** 내 문의 목록 조회 (GET /api/board/my) */
export const getMyInquiries = async (userId: string, token: string, page: number = 0, size: number = 10): Promise<Page<InquiryDto>> => {
    const response = await fetch(`${API_BASE_URL}/api/board/my?userId=${userId}&page=${page}&size=${size}`, {
        headers: getAuthHeaders(token),
    });
    if (!response.ok) throw new Error('내 문의 목록을 불러오는데 실패했습니다.');
    return response.json();
};

/** 문의 상세 조회 (GET /api/board/{id}) */
export const getInquiryDetail = async (inquiryId: number, userId: string, token: string): Promise<InquiryDetailDto> => {
    const response = await fetch(`${API_BASE_URL}/api/board/${inquiryId}?userId=${userId}`, {
        headers: getAuthHeaders(token),
    });
    if (!response.ok) throw new Error('문의 상세 정보를 불러오는데 실패했습니다.');
    return response.json();
};

/** 새 문의 작성 (POST /api/board) */
export const createInquiry = async (userId: string, data: { title: string; content: string; category: 'INQUIRY' | 'SUGGESTION' }, token: string): Promise<InquiryDto> => {
    const response = await fetch(`${API_BASE_URL}/api/board?userId=${userId}`, {
        method: 'POST',
        headers: getAuthHeaders(token),
        body: JSON.stringify(data),
    });
    if (!response.ok) throw new Error('문의 등록에 실패했습니다.');
    return response.json();
};

/** 문의 수정 (PUT /api/board/{id}) */
export const updateInquiry = async (
  inquiryId: number,
  userId: string,
  data: { title: string; content: string; category: 'INQUIRY' | 'SUGGESTION' },
  token: string
): Promise<void> => {
  const response = await fetch(`${API_BASE_URL}/api/board/${inquiryId}?userId=${userId}`, {
    method: 'PUT',
    headers: getAuthHeaders(token),
    body: JSON.stringify(data),
  });
  if (!response.ok) throw new Error('문의 수정에 실패했습니다.');
};

/** 문의 삭제 (DELETE /api/board/{id}) */
export const deleteInquiry = async (inquiryId: number, userId: string, token: string): Promise<void> => {
  const response = await fetch(`${API_BASE_URL}/api/board/${inquiryId}?userId=${userId}`, {
    method: 'DELETE',
    headers: getAuthHeaders(token),
  });
  if (!response.ok) throw new Error('문의 삭제에 실패했습니다.');
};