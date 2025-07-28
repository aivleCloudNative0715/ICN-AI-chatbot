// src/components/admin/tab/AdminContentBoard.tsx
'use client';

import React, { useState } from 'react';
import { Calendar } from 'primereact/calendar';
import { Dropdown } from 'primereact/dropdown';
import { Checkbox } from 'primereact/checkbox';
import { InputText } from 'primereact/inputtext';
import CustomPriorityDropdown from '@/components/CustomPriorityDropdown';
import { CalendarDaysIcon, TagIcon, UserIcon } from '@heroicons/react/24/outline';
// AdminAnswerBoard는 더 이상 여기서 직접 임포트하지 않습니다.

interface AdminContentBoardProps {
  type: 'dashboard' | 'inquiry' | 'suggestion' | 'pending' | 'completed';
  onSelectInquiry: (inquiry: any) => void; // 새로운 prop 추가
}

export default function AdminContentBoard({ type, onSelectInquiry }: AdminContentBoardProps) {
  // Existing states
  const [dateRange, setDateRange] = useState<[Date | null, Date | null]>([null, null]);
  const [category, setCategory] = useState('전체');
  const [priority, setPriority] = useState<string[]>([]);
  const [status, setStatus] = useState<string[]>([]);
  const [search, setSearch] = useState('');

  // Dropdown options (unchanged)
  const categoryOptions = [
    { label: '전체', value: '전체' },
    { label: '문의사항', value: '문의사항' },
    { label: '건의사항', value: '건의사항' },
  ];

  const priorityOptions = ['높음', '보통', '낮음'];
  const statusOptions = ['미처리', '완료'];

  // Dummy data (unchanged)
  const [inquiries, setInquiries] = useState([
    {
      id: 1,
      title: '문의 사항 제목입니다.',
      content: '문의 사항 내용 미리보기 입니다.',
      author: '김00',
      date: '2025-07-16',
      category: '문의사항',
      priority: '높음',
      status: '미처리',
    },
    {
      id: 2,
      title: '문의 사항 제목입니다.',
      content: '문의 사항 내용 미리보기 입니다.',
      author: 'abcd',
      date: '2025-07-16',
      category: '건의사항',
      priority: '보통',
      status: '완료',
    },
  ]);

  // 중요도 변경 핸들러 (이전과 동일)
  const handlePriorityChange = (id: number, newPriority: string) => {
    setInquiries((prevInquiries) =>
      prevInquiries.map((inquiry) =>
        inquiry.id === id ? { ...inquiry, priority: newPriority } : inquiry
      )
    );
  };

  // Handler for clicking an inquiry item
  // 이제 setSelectedInquiry 대신 onSelectInquiry를 호출합니다.
  const handleInquiryClick = (inquiry: any) => {
    onSelectInquiry(inquiry);
  };

  // AdminContentBoard는 이제 AdminAnswerBoard를 렌더링하지 않습니다.
  // 따라서 selectedInquiry 상태와 관련된 조건부 렌더링 로직이 제거됩니다.

  return (
    <div className="p-4 bg-white rounded-lg shadow-md">
      <div className="flex flex-col lg:flex-row border-2 rounded-t-[20px] border-[#C5C5C5] p-4 gap-4">
        {/* Search Input (이전과 동일) */}
        <div className="flex items-end gap-4 lg:w-2/3">
          <InputText
            placeholder="검색은 제목으로만 가능"
            className="w-full border-0 border-b-2 border-gray-300 rounded-none focus:outline-none focus:ring-0 focus:border-black"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <i className="pi pi-search font-bold" />
        </div>

        {/* Filter Area (이전과 동일) */}
        <div className='grid grid-cols-[1fr_2fr] items-center gap-2 w-full lg:w-1/3'>
          {/* Created Date */}
          <label>작성 날짜</label>
          <Calendar
            value={dateRange}
            className='border rounded-md'
            onChange={(e) => setDateRange(e.value as [Date, Date])}
            selectionMode="range"
            dateFormat="yy/mm/dd"
            readOnlyInput
            />

          {/* Category */}
          <label>카테고리</label>
          <Dropdown
            value={category}
            className='border rounded-md'
            options={categoryOptions}
            onChange={(e) => setCategory(e.value)}
          />

          {/* Priority (Filter) */}
          <span>중요도</span>
          <div className="flex gap-2">
              {priorityOptions.map((p) => (
                <div key={p} className="flex items-center gap-1">
                  <Checkbox
                    inputId={p}
                    value={p}
                    onChange={(e) => {
                      const value = e.value;
                      setPriority((prev) =>
                        prev.includes(value) ? prev.filter((x) => x !== value) : [...prev, value]
                      );
                    }}
                    checked={priority.includes(p)}
                  />
                  <label htmlFor={p}>{p}</label>
                </div>
              ))}
            </div>

            {/* Answer Processing Status */}
            <span>답변 처리</span>
            <div className="flex gap-2">
              {statusOptions.map((s) => (
                <div key={s} className="flex items-center gap-1">
                  <Checkbox
                    inputId={s}
                    value={s}
                    onChange={(e) => {
                      const value = e.value;
                      setStatus((prev) =>
                        prev.includes(value) ? prev.filter((x) => x !== value) : [...prev, value]
                      );
                    }}
                    checked={status.includes(s)}
                  />
                  <label htmlFor={s}>{s}</label>
                </div>
              ))}
            </div>
        </div>
      </div>

      {/* List (이전과 동일) */}
      <div className="space-y-4 p-4 border-2 border-t-0 rounded-b-[20px] border-[#C5C5C5]">
        {inquiries.map((item) => (
          <div
            key={item.id}
            className="border rounded-lg p-4 shadow-sm flex flex-col md:flex-row justify-between items-start md:items-center cursor-pointer hover:bg-gray-50"
            onClick={() => handleInquiryClick(item)} // 클릭 시 onSelectInquiry 호출
          >
            <div>
              <h3 className="text-lg font-bold">{item.title}</h3>
              <p className="text-gray-600">{item.content}</p>
              <div className="flex items-center gap-4 text-sm text-gray-500 mt-2">
                <div className='flex gap-2'> <UserIcon className="h-4 w-4" /> {item.author}</div>
                <div className='flex gap-2'> <CalendarDaysIcon className="h-4 w-4" /> {item.date}</div>
                <div className='flex gap-2'> <TagIcon className="h-4 w-4" /> {item.category}</div>
              </div>
            </div>
            <div className="flex items-center gap-2 mt-2 md:mt-0">
              {/* Importance Dropdown */}
              <CustomPriorityDropdown
                value={item.priority}
                onChange={(newValue) => handlePriorityChange(item.id, newValue)}
              />
              <span
                className={`px-2 py-1 text-sm rounded border ${
                  item.status === '완료'
                    ? 'border-green-500 text-green-600'
                    : 'border-gray-500 text-gray-600'
                }`}
              >
                {item.status}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Pagination (이전과 동일) */}
      <div className="flex justify-center items-center mt-6 gap-2">
        {[1, 2, 3, 4, 5].map((num) => (
          <button
            key={num}
            className="px-3 py-1 border rounded hover:bg-gray-100"
          >
            {num}
          </button>
        ))}
        <span>...</span>
        <button className="px-3 py-1 border rounded hover:bg-gray-100">10</button>
      </div>
    </div>
  );
}