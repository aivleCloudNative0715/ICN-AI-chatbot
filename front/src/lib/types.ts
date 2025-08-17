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

// Enum 타입들 (백엔드와 일치시킴)
export type InquiryStatus = 'PENDING' | 'RESOLVED';
export type Urgency = 'HIGH' | 'MEDIUM' | 'LOW';
export type BoardCategory = 'INQUIRY' | 'SUGGESTION';
export type AdminRole = 'ADMIN' | 'SUPER';


// --- 관리자 페이지용 타입 (신규 추가 및 수정) ---

// 관리자 페이지 문의 목록에서 사용할 타입 (AdminInquiryService.getInquiries 참고)
export interface AdminInquiryDto {
  inquiryId: number;
  title: string;
  userId: string;
  createdAt: string; // ISO 날짜 문자열
  status: InquiryStatus;
  urgency: Urgency;
  category: BoardCategory;
}

// 관리자 페이지 문의 상세 조회 시 사용할 타입 (AdminInquiryService.getInquiryDetail 참고)
export interface AdminInquiryDetailDto extends AdminInquiryDto {
  content: string;
  answer?: string; // 답변은 없을 수도 있으므로 optional
  adminId?: string;
  updatedAt: string;
}

// 관리자 계정 정보 타입 (AdminService.getAdmins 참고)
export interface AdminDto {
  id: number;
  adminId: string;
  adminName: string;
  role: AdminRole;
  createdAt: string;
}

// 대시보드 문의 건수 타입
export interface InquiryCounts {
    total: number;
    pending: number;
    resolved: number;
    inquiry: number;
    suggestion: number;
    high: number;
    medium: number;
    low: number;
}

// =======================================================
//                공항 정보 API 타입
// =======================================================

export interface ParkingInfo {
  floor: string;
  parking: string;
  parkingarea: string;
  datetm: string;
}

export interface PassengerForecast {
  adate: string;
  atime: string;
  
  // T1 입국장 (개별) - 필드 추가
  t1sum1: number;
  t1sum2: number;
  t1sum3: number;
  t1sum4: number;

  // T1 출국장 (개별) - 필드 추가
  t1sum5: number;
  t1sum6: number;
  t1sum7: number;
  t1sum8: number;

  // T1 합계
  t1sumset1: number;
  t1sumset2: number;

  // T2 입국장 (개별) - 필드 추가
  t2sum1: number;
  t2sum2: number;

  // T2 출국장 (개별) - 필드 추가
  t2sum3: number;
  t2sum4: number;

  // T2 합계
  t2sumset1: number;
  t2sumset2: number;
}

export interface FlightArrival {
  airline: string;
  airport: string;
  airportCode: string;
  carousel: string;
  estimatedDateTime: string;
  exitnumber: string;
  flightId: string;
  gatenumber: string;
  remark: string;
  scheduleDateTime: string;
  terminalid: string;
}

export interface FlightDeparture {
  airline: string;
  airport: string;
  airportCode: string;
  chkinrange: string;
  estimatedDateTime: string;
  flightId: string;
  gatenumber: string;
  remark: string;
  scheduleDateTime: string;
  terminalid: string;
}

export interface ArrivalWeatherInfo {
  temp: string; // 관측 기온
  wimage: string; // 날씨 이미지 URL
}