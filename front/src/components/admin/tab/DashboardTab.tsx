'use client';

import React from 'react';

export default function DashboardTab() {
  return (
    <div className="p-4 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">대시보드 개요</h2>
      {/* 대시보드 요약 정보 */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-8 text-center">
        <div className="p-4 border rounded-lg bg-blue-100">
          <p className="text-lg font-semibold">전체</p>
          <p className="text-2xl font-bold">80</p>
        </div>
        <div className="p-4 border rounded-lg bg-blue-100">
          <p className="text-lg font-semibold">문의 사항</p>
          <p className="text-2xl font-bold">80</p>
        </div>
        <div className="p-4 border rounded-lg bg-blue-100">
          <p className="text-lg font-semibold">건의 사항</p>
          <p className="text-2xl font-bold">80</p>
        </div>
        <div className="p-4 border rounded-lg bg-blue-100">
          <p className="text-lg font-semibold">미처리</p>
          <p className="text-2xl font-bold">80</p>
        </div>
        <div className="p-4 border rounded-lg bg-blue-100">
          <p className="text-lg font-semibold">완료</p>
          <p className="text-2xl font-bold">80</p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8 text-center">
        <div className="p-4 border rounded-lg bg-orange-100">
          <p className="text-lg font-semibold">높음</p>
          <p className="text-2xl font-bold">80</p>
        </div>
        <div className="p-4 border rounded-lg bg-green-100">
          <p className="text-lg font-semibold">보통</p>
          <p className="text-2xl font-bold">80</p>
        </div>
        <div className="p-4 border rounded-lg bg-red-100">
          <p className="text-lg font-semibold">낮음</p>
          <p className="text-2xl font-bold">80</p>
        </div>
      </div>

      {/* 검색 및 필터링 영역 */}
      <div className="mb-8 p-4 border rounded-lg bg-gray-50">
        <div className="flex flex-col md:flex-row justify-between items-center mb-4 space-y-4 md:space-y-0 md:space-x-4">
          <div className="flex items-center space-x-4 w-full md:w-auto">
            <label htmlFor="date-range" className="text-gray-700 whitespace-nowrap">작성 날짜</label>
            <input
              type="text"
              id="date-range"
              defaultValue="2025/07/16 - 2025/07/17"
              className="p-2 border rounded-md w-full"
            />
          </div>
          <div className="flex items-center space-x-4 w-full md:w-auto">
            <label htmlFor="category" className="text-gray-700 whitespace-nowrap">카테고리</label>
            <select id="category" className="p-2 border rounded-md w-full">
              <option>전체</option>
            </select>
          </div>
        </div>
        
        <div className="flex flex-col sm:flex-row items-start sm:items-center space-y-4 sm:space-y-0 sm:space-x-8 mb-4">
          <span className="text-gray-700 whitespace-nowrap">중요도</span>
          <label className="inline-flex items-center">
            <input type="checkbox" className="form-checkbox h-5 w-5 text-blue-600 rounded" />
            <span className="ml-2 text-gray-700">높음</span>
          </label>
          <label className="inline-flex items-center">
            <input type="checkbox" className="form-checkbox h-5 w-5 text-blue-600 rounded" />
            <span className="ml-2 text-gray-700">보통</span>
          </label>
          <label className="inline-flex items-center">
            <input type="checkbox" className="form-checkbox h-5 w-5 text-blue-600 rounded" />
            <span className="ml-2 text-gray-700">낮음</span>
          </label>
        </div>

        <div className="flex flex-col sm:flex-row items-start sm:items-center space-y-4 sm:space-y-0 sm:space-x-8">
          <span className="text-gray-700 whitespace-nowrap">답변 처리</span>
          <label className="inline-flex items-center">
            <input type="checkbox" className="form-checkbox h-5 w-5 text-blue-600 rounded" />
            <span className="ml-2 text-gray-700">미처리</span>
          </label>
          <label className="inline-flex items-center">
            <input type="checkbox" className="form-checkbox h-5 w-5 text-blue-600 rounded" />
            <span className="ml-2 text-gray-700">완료</span>
          </label>
        </div>

        <div className="flex items-center mt-4">
          <input
            type="text"
            placeholder="검색은 제목으로만 가능"
            className="flex-grow p-2 border rounded-l-md"
          />
          <button className="p-2 border border-l-0 rounded-r-md bg-white hover:bg-gray-100">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </button>
        </div>
      </div>

      <p className="text-gray-700">관리자 대시보드의 실제 목록 콘텐츠를 여기에 구현하세요.</p>
      <p className="text-gray-500 mt-4">예: 최근 문의 목록, 차트 등</p>
    </div>
  );
}
