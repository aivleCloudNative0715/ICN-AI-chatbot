// src/lib/api.ts
import { InquiryDto, InquiryDetailDto } from './types';

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;

// JWT 토큰 가져오기 (실제로는 Context 등에서 관리)
const getAuthHeaders = () => {
  const token = localStorage.getItem('jwt_token');
  return {
    'Content-Type': 'application/json',
    ...(token && { 'Authorization': `Bearer ${token}` }),
  };
};

/** 전체 게시글 목록 조회 (GET /api/board) */
export const getAllInquiries = async (page: number = 0, size: number = 10): Promise<Page<InquiryDto>> => {
  const response = await fetch(`${API_BASE_URL}/api/board?page=${page}&size=${size}`, {
    headers: getAuthHeaders(),
  });
  if (!response.ok) throw new Error('게시글 목록을 불러오는데 실패했습니다.');
  return response.json();
};

/** 내 문의 목록 조회 (GET /api/board/my) */
export const getMyInquiries = async (userId: string, page: number = 0, size: number = 10): Promise<Page<InquiryDto>> => {
    const response = await fetch(`${API_BASE_URL}/api/board/my?userId=${userId}&page=${page}&size=${size}`, {
        headers: getAuthHeaders(),
    });
    if (!response.ok) throw new Error('내 문의 목록을 불러오는데 실패했습니다.');
    return response.json();
};

/** 문의 상세 조회 (GET /api/board/{id}) */
export const getInquiryDetail = async (inquiryId: number, userId: string): Promise<InquiryDetailDto> => {
    // 백엔드의 getMyInquiryDetail을 활용
    const response = await fetch(`${API_BASE_URL}/api/board/${inquiryId}?userId=${userId}`, {
        headers: getAuthHeaders(),
    });
    if (!response.ok) throw new Error('문의 상세 정보를 불러오는데 실패했습니다.');
    return response.json();
};

/** 새 문의 작성 (POST /api/board) */
export const createInquiry = async (userId: string, data: { title: string; content: string; category: 'INQUIRY' | 'SUGGESTION' }): Promise<InquiryDto> => {
    const response = await fetch(`${API_BASE_URL}/api/board?userId=${userId}`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify(data),
    });
    if (!response.ok) throw new Error('문의 등록에 실패했습니다.');
    return response.json();
};

// 여기에 문의 수정, 삭제 API 호출 함수를 추가...

// Page 타입을 정의해줍니다.
export interface Page<T> {
  content: T[];
  totalPages: number;
  totalElements: number;
  // ... 기타 페이징 정보
}

/** 문의 수정 (PUT /api/board/{id}) */
export const updateInquiry = async (
  inquiryId: number,
  userId: string,
  data: { title: string; content: string; category: 'INQUIRY' | 'SUGGESTION' }
): Promise<InquiryDto> => {
  const response = await fetch(`${API_BASE_URL}/api/board/${inquiryId}?userId=${userId}`, {
    method: 'PUT', // PUT 메서드 사용
    headers: getAuthHeaders(),
    body: JSON.stringify(data),
  });
  if (!response.ok) throw new Error('문의 수정에 실패했습니다.');
  // 백엔드에서 수정된 문의 정보를 반환한다고 가정
  return response.json();
};

/** 문의 삭제 (DELETE /api/board/{id}) */
export const deleteInquiry = async (inquiryId: number, userId: string): Promise<void> => {
  const response = await fetch(`${API_BASE_URL}/api/board/${inquiryId}?userId=${userId}`, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  });
  if (!response.ok) throw new Error('문의 삭제에 실패했습니다.');
  // 성공 시 내용 없이 2xx 상태 코드를 반환하므로, json() 호출은 필요 없음
};