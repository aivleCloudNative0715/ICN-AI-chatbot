'use client';

import React, { useState } from 'react';
import { FileUpload } from 'primereact/fileupload';
import { DataTable } from 'primereact/datatable';
import { Column } from 'primereact/column';
import { ExclamationTriangleIcon } from '@heroicons/react/24/outline';

interface UploadedFile {
  name: string;
  size: number;
  type: string;
}

export default function AdminFileUploadTab() {
  const [files, setFiles] = useState<UploadedFile[]>([]);

  const onUpload = (event: any) => {
    const uploadedFiles = event.files.map((file: File) => ({
      name: file.name,
      size: file.size,
      type: file.type,
    }));
    setFiles((prev) => [...prev, ...uploadedFiles]);
  };

  const sizeTemplate = (rowData: UploadedFile) => `${(rowData.size / 1024).toFixed(2)} KB`;

  const deleteTemplate = (rowData: UploadedFile) => (
    <button
      className="text-red-500 hover:text-red-700 font-bold"
      onClick={() => setFiles((prev) => prev.filter((f) => f.name !== rowData.name))}
    >
      삭제
    </button>
  );

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
      <div className="mb-6 border border-gray-400 rounded-lg p-2 bg-gray-50 text-center">
        <FileUpload
          name="files[]"
          url="/api/upload"
          customUpload
          multiple
          uploadHandler={onUpload}
          accept=".pdf,.docx,.hwp,.xlsx"
          maxFileSize={5 * 1024 * 1024}
          chooseOptions={{ label: '파일 추가', className: 'bg-black text-white px-4 py-2 rounded hover:bg-gray-800' }}
          uploadOptions={{ label: '파일 업로드', className: 'border border-gray-400 px-4 py-2 rounded hover:bg-black hover:text-white transition' }}
          cancelOptions={{ label: '취소', className: 'border border-gray-400 px-4 py-2 rounded hover:bg-gray-200 transition' }}
          emptyTemplate={<p className="m-0 text-gray-500">파일을 드래그하거나 선택하세요</p>}
        />
      </div>

      {/* 업로드된 파일 목록 */}
      <div className="bg-gray-50 p-4 rounded-lg mb-6">
        <h3 className="text-lg font-semibold mb-4">업로드된 파일 목록</h3>
        <DataTable value={files} emptyMessage="업로드된 파일이 없습니다." responsiveLayout="scroll">
          <Column field="name" header="파일명" />
          <Column field="type" header="유형" />
          <Column field="size" header="크기" body={sizeTemplate} />
          <Column body={deleteTemplate} header="삭제" style={{ width: '100px' }} />
        </DataTable>
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
