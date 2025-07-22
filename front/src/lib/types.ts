// src/lib/types.ts

// 문의/건의 기본 타입
export interface Inquiry {
  inquiry_id: string;
  user_id: string; // 실제로는 JWT 토큰으로 인증하지만, 데이터 구조상 필요
  title: string;
  content: string;
  category: '문의' | '건의';
  urgency: '높음' | '보통' | '낮음';
  status: '미처리' | '답변처리 완료';
  created_at: string; // ISO 8601 string (e.g., "2023-07-20T10:00:00Z")
  updated_at: string; // ISO 8601 string
  is_deleted: boolean;
}

// 답변 타입
export interface InquiryAnswer {
  answer_id: string;
  inquiry_id: string;
  user_id: string; // 답변 작성자 (문의 사용자)
  admin_id: string; // 답변 처리 관리자
  content: string;
  created_at: string;
  updated_at: string;
}

// 게시판 목록에 사용될 Inquiry (부분적으로)
export interface InquiryListItem extends Inquiry {
  // 목록에서 필요한 추가 정보 또는 answer 존재 여부
  hasAnswer: boolean;
  answerContentPreview?: string; // 답변 내용 미리보기
}