// src/lib/types.ts
export interface InquiryDto {
  inquiryId: number;
  userId: string;
  title: string;
  category: 'INQUIRY' | 'SUGGESTION'; // 백엔드 Enum과 동일하게
  urgency: 'HIGH' | 'MEDIUM' | 'LOW';
  status: 'PENDING' | 'RESOLVED';
  createdAt: string; // LocalDateTime은 string으로 받습니다.
  updatedAt: string;
}

// 상세 조회 시 사용될 타입 (답변 포함)
export interface InquiryDetailDto extends InquiryDto {
  content: string;
  answer?: string;
  adminId?: string;
}

// 사이드바에서 사용할 타입
export type PostCategory = 'inquiry' | 'suggestion';
export type PostFilter = 'all' | 'my';

/**
 * 백엔드 API의 페이지네이션 응답을 위한 제네릭 타입
 * T는 페이지의 content에 들어갈 데이터의 타입을 의미합니다. (예: InquiryDto)
 */
export interface Page<T> {
  content: T[];          // 현재 페이지의 데이터 목록
  totalPages: number;      // 총 페이지 수
  totalElements: number;   // 모든 페이지의 총 요소 수
  size: number;            // 한 페이지의 크기
  number: number;          // 현재 페이지 번호 (0부터 시작)
  first: boolean;          // 첫 페이지 여부
  last: boolean;           // 마지막 페이지 여부
  empty: boolean;          // 현재 페이지가 비어있는지 여부
}