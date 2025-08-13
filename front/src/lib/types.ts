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