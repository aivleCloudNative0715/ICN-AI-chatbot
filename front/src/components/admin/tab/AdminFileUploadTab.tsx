// src/components/admin/tab/AdminFileUploadTab.tsx
'use client';

import { useState } from 'react';
import { FileUpload, FileUploadHandlerEvent } from 'primereact/fileupload';
import { Dropdown } from 'primereact/dropdown';
import { useAuth } from '@/contexts/AuthContext';
import { ProgressBar } from 'primereact/progressbar';
import { Tag } from 'primereact/tag';
import { ExclamationTriangleIcon } from '@heroicons/react/24/outline';

export default function AdminFileUploadTab() {
  const { token } = useAuth();
  
  // 1. 업로드할 파일의 카테고리를 관리할 상태를 추가합니다.
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [totalSize, setTotalSize] = useState(0);
  
  // 카테고리 선택 옵션을 정의합니다. (필요에 따라 추가/수정)
  const categoryOptions = [
    { label: '공항 정보 (Airport Info)', value: 'airport_info' },
    { label: '항공사 정보 (Airline Info)', value: 'airline_info' },
    { label: '시설 정보 (Facility Info)', value: 'facility_info' },
    { label: '환승 소요시간 (Connection Time)', value: 'connection_time' },
    { label: '환승 경로 (Transit Path)', value: 'transit_path' },
    { label: '주차장 정보 (Parking Lot)', value: 'parking_lot' },
    { label: '주차장 정책 (Parking Lot Policy)', value: 'parking_lot_policy' },
    { label: '공항 정책 (Airport Policy)', value: 'airport_policy' },
  ];

  // 2. 파일 업로드를 처리하는 핸들러 함수를 새로 작성합니다.
  const uploadHandler = async (event: FileUploadHandlerEvent) => {
    if (!selectedCategory) {
      alert('파일 카테고리를 먼저 선택해주세요.');
      return; // 카테고리가 없으면 업로드 중지
    }
    if (!token) {
        alert('인증 정보가 없습니다. 다시 로그인해주세요.');
        return;
    }

    const aiServerUrl = process.env.NEXT_PUBLIC_AI_SERVER_URL;
    const files = event.files;

    // FormData 객체를 생성하여 서버로 보낼 데이터를 담습니다.
    const formData = new FormData();
    formData.append('category', selectedCategory);
    
    files.forEach((file) => {
      formData.append('file', file, file.name);
    });

    try {
      // 3. fetch를 사용하여 AI 서버로 FormData를 전송합니다.
      const response = await fetch(`${aiServerUrl}/chatbot/upload`, {
        method: 'POST',
        headers: {
          // FormData를 전송할 때는 'Content-Type'을 설정하지 않습니다.
          // 브라우저가 자동으로 'multipart/form-data'와 경계(boundary)를 설정해줍니다.
          'Authorization': `Bearer ${token}`, // AI 서버에 인증이 필요하다면 토큰을 추가합니다.
        },
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || '파일 업로드에 실패했습니다.');
      }
      
      // PrimeReact의 FileUpload 컴포넌트에 성공을 알립니다.
      event.options.clear(); // 성공 시 업로드 목록 초기화
      alert('파일이 성공적으로 업로드되었습니다.');

    } catch (error) {
      // PrimeReact의 FileUpload 컴포넌트에 실패를 알립니다.
      console.error("Upload error: ", error);
      alert(error instanceof Error ? error.message : '업로드 중 오류가 발생했습니다.');
    }
  };

  return (
    <div className="p-4 bg-white rounded-lg shadow-md">
      {/* 안내 박스 */}
      <div className="bg-white p-6 rounded-lg border border-gray-300 mb-6">
        <p className="text-gray-700 mb-2 leading-relaxed">
          정확하고 풍부한 답변을 위한 지식 자료를 추가해 주세요.<br />
          PDF, 한글, 워드 등 다양한 형식의 규정집, 안내서 등을 업로드하시면 챗봇이 이를 학습하여 답변 품질을 높입니다.
        </p>
        <div className="flex items-center text-yellow-600 font-semibold mb-3">
          <ExclamationTriangleIcon className="w-5 h-5 mr-2" />
          <span>
            중요 안내 - <span className="text-red-500 font-bold">민감 정보 포함 금지</span>
          </span>
        </div>
        <p className="text-gray-700 mb-4">
          해당 자료는 AI 시스템의 지식 기반(<span className="font-bold">RAG, Retrieval-Augmented Generation</span>)에
          사용되어 사용자 질의에 대한 답변을 생성합니다.
        </p>
        <p className="text-gray-700 font-semibold mb-2">따라서, 문서 제공 시 다음 사항을 꼭 확인해 주시기 바랍니다:</p>
        <ul className="list-disc list-inside text-gray-700 space-y-1">
          <li>
            <span className="font-bold">
              개인 식별 정보(PII), 내부 정책, 대외비, 보안 정보는{' '}
              <span className="text-red-500 font-bold">절대</span> 포함하지 말아 주세요.
            </span>
          </li>
          <li>문서는 반드시 외부에 공개 가능한 범위로만 제공해 주세요.</li>
        </ul>
      </div>

      {/* 업로드 박스 */}
      <div className="p-4 sm:p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">학습 데이터 업로드</h2>
      
      {/* 카테고리 선택 Dropdown */}
      <div className="mb-6">
        <label htmlFor="category" className="block text-lg font-semibold text-gray-700 mb-2">
          파일 카테고리 선택
        </label>
        <Dropdown
          id="category"
          value={selectedCategory}
          options={categoryOptions}
          onChange={(e) => setSelectedCategory(e.value)}
          placeholder="업로드할 파일의 종류를 선택하세요"
          className="w-full md:w-1/2 border-2"
        />
      </div>

      {/* 파일 업로드 컴포넌트 */}
      <FileUpload
        name="files[]"
        customUpload // 4. customUpload 모드를 활성화하여 직접 만든 핸들러를 사용합니다.
        uploadHandler={uploadHandler}
        multiple
        accept="application/pdf" // .pdf 파일만 받도록 설정
        maxFileSize={10000000} // 최대 파일 크기 (예: 10MB)
        emptyTemplate={<p className="m-0">여기에 파일을 드래그하거나 선택하여 업로드하세요.</p>}
        chooseLabel="파일 선택"
        uploadLabel="업로드"
        cancelLabel="취소"
        // 5. 선택된 파일이 없을 때 업로드 버튼이 비활성화되도록 설정
        disabled={!selectedCategory}
      />
    </div>

      {/* 참고 데이터 링크 */}
      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <h4 className="text-xl mb-4">📂 현재 제공하고 있는 파일/API 데이터 입니다.</h4>

        <h5 className="text-lg font-semibold mb-2">파일 데이터</h5>
        <ul className="list-disc list-inside mb-4 text-blue-600 space-y-1">
          <li><a href="https://www.data.go.kr/data/15048966/fileData.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_환승승객현황 정보</a></li>
          <li><a href="https://www.data.go.kr/data/15037386/fileData.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_출국장도보소요시간정보</a></li>
          <li><a href="https://www.data.go.kr/data/15063436/fileData.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_주차장별 도보소요시간정보</a></li>
          <li><a href="https://www.data.go.kr/data/15070832/fileData.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_승용차 정차구역별 도보 소요시간 정보</a></li>
          <li><a href="https://www.data.go.kr/data/15102222/fileData.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_공항철도-체크인카운터 간 도보소요시간 정보</a></li>
        </ul>

        <h5 className="text-lg font-semibold mb-2">API 활용데이터</h5>
        <ul className="list-disc list-inside text-blue-600 space-y-1">
          <li><a href="https://www.data.go.kr/data/15095086/openapi.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_기상 정보</a></li>
          <li><a href="https://www.data.go.kr/data/15095066/openapi.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_승객예고-출·입국장별</a></li>
          <li><a href="https://www.data.go.kr/data/15095061/openapi.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_입국장현황 정보 서비스</a></li>
          <li><a href="https://www.data.go.kr/data/15095073/openapi.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_취항 항공사 현황 조회</a></li>
          <li><a href="https://www.data.go.kr/tcs/dss/selectApiDataDetailView.do?publicDataPk=15140153" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_항공기 운항 현황 상세 조회</a></li>
          <li><a href="https://www.data.go.kr/data/15095093/openapi.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_여객편 운항현황(다국어)</a></li>
          <li><a href="https://www.data.go.kr/data/15134279/openapi.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_여객기 운항편 조회(면세점용)</a></li>
          <li><a href="https://www.data.go.kr/data/15112968/openapi.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_여객기 운항 현황 상세 조회 서비스</a></li>
          <li><a href="https://www.data.go.kr/data/15095074/openapi.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_여객편 주간 운항 현황</a></li>
          <li><a href="https://www.data.go.kr/data/15095059/openapi.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_여객기 정기운항편 일정 정보</a></li>
          <li><a href="https://www.data.go.kr/data/15114085/openapi.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_여객기 정기운항편 현황 상세 조회 서비스</a></li>
          <li><a href="https://www.data.go.kr/data/15134281/openapi.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_여객기 정기운항편 조회(면세점용)</a></li>
          <li><a href="https://www.data.go.kr/data/15095051/openapi.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_화물기 정기운항편 일정 정보</a></li>
          <li><a href="https://www.data.go.kr/data/15113461/openapi.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_화물기 운항 현황 상세 조회 서비스</a></li>
          <li><a href="https://www.data.go.kr/data/15114086/openapi.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_화물기 정기운항편 현황 상세 조회 서비스</a></li>
          <li><a href="https://www.data.go.kr/data/15095047/openapi.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_주차 정보</a></li>
          <li><a href="https://www.data.go.kr/data/15095053/openapi.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_주차장별 요금 부과 기준 정보</a></li>
          <li><a href="https://www.data.go.kr/data/15107228/openapi.do" target="_blank" rel="noopener noreferrer" className="hover:underline">인천국제공항공사_주차면 현황 정보</a></li>
        </ul>
      </div>
    </div>
  );
}
