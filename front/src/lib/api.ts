// src/lib/api.ts
import { 
  AdminInquiryDto, 
  AdminInquiryDetailDto, 
  InquiryStatus, 
  Urgency, 
  BoardCategory,
  InquiryCounts,
  AdminDto,
  InquiryDto,
  InquiryDetailDto,
  Page
} from './types';

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;

const getAuthHeaders = (token: string | null) => {
  return {
    'Content-Type': 'application/json',
    ...(token && { 'Authorization': `Bearer ${token}` }),
  };
};

/** 전체 게시글 목록 조회 (GET /api/board) */
export const getAllInquiries = async (
  category: 'INQUIRY' | 'SUGGESTION', // 카테고리 타입을 명확히 합니다.
  token: string,                      // 토큰을 필수로 받습니다.
  page: number = 0, 
  size: number = 10,
  search: string = ''
): Promise<Page<InquiryDto>> => {
  const searchParam = search ? `&search=${encodeURIComponent(search)}` : '';
  const response = await fetch(`${API_BASE_URL}/api/board?category=${category}&page=${page}&size=${size}${searchParam}`, {
    headers: getAuthHeaders(token),
  });
  if (!response.ok) throw new Error('게시글 목록을 불러오는데 실패했습니다.');
  return response.json();
};

/** 내 문의 목록 조회 (GET /api/board/my) */
export const getMyInquiries = async (
  category: 'INQUIRY' | 'SUGGESTION' | null,
  token: string, 
  page: number = 0, 
  size: number = 10,
  search: string = ''
): Promise<Page<InquiryDto>> => {
    const searchParam = search ? `&search=${encodeURIComponent(search)}` : '';
  const response = await fetch(`${API_BASE_URL}/api/board/my?category=${category}&page=${page}&size=${size}${searchParam}`, {
    headers: getAuthHeaders(token),
  });
    if (!response.ok) throw new Error('내 문의 목록을 불러오는데 실패했습니다.');
    return response.json();
};

/** 문의 상세 조회 (GET /api/board/{id}) */
export const getInquiryDetail = async (inquiryId: number, token: string): Promise<InquiryDetailDto> => {
    // userId 파라미터를 URL에서 제거
    const response = await fetch(`${API_BASE_URL}/api/board/${inquiryId}`, {
        headers: getAuthHeaders(token),
    });
    if (!response.ok) throw new Error('문의 상세 정보를 불러오는데 실패했습니다.');
    return response.json();
};

/** 새 문의 작성 (POST /api/board) */
export const createInquiry = async (data: { title: string; content: string; category: 'INQUIRY' | 'SUGGESTION' }, token: string): Promise<InquiryDto> => {
    // userId 파라미터를 URL에서 제거
    const response = await fetch(`${API_BASE_URL}/api/board`, {
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
  data: { title: string; content: string; category: 'INQUIRY' | 'SUGGESTION' },
  token: string
): Promise<void> => {
  const response = await fetch(`${API_BASE_URL}/api/board/${inquiryId}`, {
    method: 'PUT',
    headers: getAuthHeaders(token),
    body: JSON.stringify(data),
  });
  if (!response.ok) throw new Error('문의 수정에 실패했습니다.');
};

/** 문의 삭제 (DELETE /api/board/{id}) */
export const deleteInquiry = async (inquiryId: number, token: string): Promise<void> => {
  const response = await fetch(`${API_BASE_URL}/api/board/${inquiryId}`, {
    method: 'DELETE',
    headers: getAuthHeaders(token),
  });
  if (!response.ok) throw new Error('문의 삭제에 실패했습니다.');
};

// =======================================================
//                관리자 API 함수들
// =======================================================

// 관리자용 API 기본 URL
const ADMIN_API_URL = `${API_BASE_URL}/admin`;

/**
 * [관리자] 문의 목록 조회 (GET /admin/inquiries)
 */
export const getAdminInquiries = async (
  token: string,
  page: number,
  size: number,
  filters: {
    status?: InquiryStatus;
    urgency?: Urgency;
    category?: BoardCategory;
    search?: string;
  }
): Promise<Page<AdminInquiryDto>> => {
  const params = new URLSearchParams({
    page: String(page),
    size: String(size),
  });
  if (filters.status) params.append('status', filters.status);
  if (filters.urgency) params.append('urgency', filters.urgency);
  if (filters.category) params.append('category', filters.category);
  if (filters.search) params.append('search', filters.search);

  const response = await fetch(`${ADMIN_API_URL}/inquiries?${params.toString()}`, {
    headers: getAuthHeaders(token),
  });
  if (!response.ok) throw new Error('관리자 문의 목록 조회에 실패했습니다.');
  return response.json();
};

/**
 * [관리자] 문의 상세 조회 (GET /admin/inquiries/{inquiry_id})
 */
export const getAdminInquiryDetail = async (inquiryId: number, token: string): Promise<AdminInquiryDetailDto> => {
    const response = await fetch(`${ADMIN_API_URL}/inquiries/${inquiryId}`, {
        headers: getAuthHeaders(token),
    });
    if (!response.ok) throw new Error('관리자 문의 상세 조회에 실패했습니다.');
    return response.json();
};


/**
 * [관리자] 문의 답변 등록/수정 (POST /admin/inquiries/{inquiry_id}/answer)
 */
export const processAdminAnswer = async (
  inquiryId: number, 
  token: string,
  payload: { adminId: string; content: string }
): Promise<AdminInquiryDetailDto> => {
  const response = await fetch(`${ADMIN_API_URL}/inquiries/${inquiryId}/answer`, {
    method: 'POST',
    headers: getAuthHeaders(token),
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error('답변 등록에 실패했습니다.');
  return response.json();
};

/**
 * [관리자] 문의 긴급도 수정 (PATCH /admin/inquiries/{inquiry_id}/urgency)
 */
export const updateInquiryUrgency = async (
  inquiryId: number,
  token: string,
  urgency: Urgency
): Promise<{ message: string }> => { // ApiMessage DTO에 해당
    const response = await fetch(`${ADMIN_API_URL}/inquiries/${inquiryId}/urgency`, {
        method: 'PATCH',
        headers: getAuthHeaders(token),
        body: JSON.stringify({ urgency }),
    });
    if (!response.ok) throw new Error('긴급도 수정에 실패했습니다.');
    return response.json();
}

/**
 * [관리자] 대시보드 문의 건수 조회 (GET /admin/inquiries/counts)
 */
export const getInquiryCounts = async (
  token: string, 
  start: string, // YYYY-MM-DDTHH:mm:ss 형식
  end: string
): Promise<InquiryCounts> => {
  const params = new URLSearchParams({
    created_at_start: start,
    created_at_end: end,
  });
  const response = await fetch(`${ADMIN_API_URL}/inquiries/counts?${params.toString()}`, {
    headers: getAuthHeaders(token),
  });
  if (!response.ok) throw new Error('문의 건수 조회에 실패했습니다.');
  return response.json();
};

/**
 * [관리자] 관리자 계정 목록 조회 (GET /admin/users)
 */
export const getAdmins = async (token: string, page: number, size: number, isActive: boolean): Promise<Page<AdminDto>> => {
  const params = new URLSearchParams({
    page: String(page),
    size: String(size),
    is_active: String(isActive),
  });
  const response = await fetch(`${ADMIN_API_URL}/users?${params.toString()}`, {
    headers: getAuthHeaders(token),
  });
  if (!response.ok) throw new Error('관리자 목록 조회에 실패했습니다.');
  return response.json();
};

/**
 * [관리자] 관리자 계정 추가 (POST /admin/users)
 */
export const addAdmin = async (token: string, data: any): Promise<AdminDto> => {
  const response = await fetch(`${ADMIN_API_URL}/users`, {
    method: 'POST',
    headers: getAuthHeaders(token),
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.message || '관리자 추가에 실패했습니다.');
  }
  return response.json();
};