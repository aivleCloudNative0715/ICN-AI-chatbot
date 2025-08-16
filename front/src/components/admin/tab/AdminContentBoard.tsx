// src/components/admin/tab/AdminContentBoard.tsx
'use client';

import React, { useState } from 'react';
import { Calendar } from 'primereact/calendar';
import { Dropdown } from 'primereact/dropdown';
import { MultiSelect } from 'primereact/multiselect';
import { InputText } from 'primereact/inputtext';
import { Paginator, PaginatorPageChangeEvent } from 'primereact/paginator';
import CustomPriorityDropdown from '@/components/CustomPriorityDropdown';
import { CalendarDaysIcon, TagIcon, UserIcon } from '@heroicons/react/24/outline';
import { useAuth } from '@/contexts/AuthContext';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { getAdminInquiries, updateInquiryUrgency } from '@/lib/api';
import { AdminInquiryDto, Urgency, BoardCategory, InquiryStatus } from '@/lib/types';
import { Nullable } from 'primereact/ts-helpers';
import { Button } from 'primereact/button';
import { useDebounce } from '@/hooks/useDebounce';

interface AdminContentBoardProps {
  type: 'dashboard' | 'inquiry' | 'suggestion' | 'pending' | 'completed';
  onSelectInquiry: (inquiry: AdminInquiryDto) => void;
}

export default function AdminContentBoard({ type, onSelectInquiry }: AdminContentBoardProps) {
  const { token } = useAuth();
  const queryClient = useQueryClient();

  const [first, setFirst] = useState(0);
  const [rows, setRows] = useState(10);
  
  // --- 상태 관리 수정 ---
  // 1. 달력 UI와 직접 바인딩될 임시 상태입니다.
  const [calendarValue, setCalendarValue] = useState<Nullable<(Date | null)[]>>(null);
  // 2. API 필터링에 실제 사용될 상태입니다. 이 값이 변경될 때만 API가 다시 호출됩니다.
  const [dateRangeFilter, setDateRangeFilter] = useState<Nullable<(Date | null)[]>>(null);
  
  const [category, setCategory] = useState<BoardCategory | null>(null);
  const [urgencies, setUrgencies] = useState<Urgency[]>([]);
  const [status, setStatus] = useState<InquiryStatus | null>(null);
  const [searchTerm, setSearchTerm] = useState('');

  const debouncedSearchTerm = useDebounce(searchTerm, 500);

  const categoryOptions = [
    { label: '전체', value: null },
    { label: '문의사항', value: 'INQUIRY' },
    { label: '건의사항', value: 'SUGGESTION' },
  ];
  const priorityOptions = [
    { label: '높음', value: 'HIGH' },
    { label: '보통', value: 'MEDIUM' },
    { label: '낮음', value: 'LOW' },
  ];
  const statusOptions = [
    { label: '전체', value: null },
    { label: '미처리', value: 'PENDING' },
    { label: '완료', value: 'RESOLVED' },
  ];
  const page = first / rows;

  const formatDate = (date: Date | null) => {
    if (!date) return undefined;
    return date.toISOString().split('T')[0]; 
  };

  const { data: inquiriesData, isLoading, isError } = useQuery({
    queryKey: ['adminInquiries', page, rows, debouncedSearchTerm, category, urgencies, status, dateRangeFilter],
    queryFn: () => {
      if (!token) return;
      return getAdminInquiries(token, page, rows, {
        search: debouncedSearchTerm || undefined,
        category: category || undefined,
        urgencies: urgencies.length > 0 ? urgencies : undefined,
        status: status || undefined,
        start: formatDate(dateRangeFilter?.[0] ?? null),
        end: formatDate(dateRangeFilter?.[1] ?? null),
      });
    },
    enabled: !!token,
  });

  const updateUrgencyMutation = useMutation({
    mutationFn: ({ inquiryId, newPriority }: { inquiryId: number, newPriority: Urgency }) => {
      if (!token) throw new Error("인증되지 않았습니다.");
      return updateInquiryUrgency(inquiryId, token, newPriority);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['adminInquiries'] });
    },
    onError: (error) => alert(`긴급도 수정 실패: ${error.message}`),
  });

  const handlePriorityChange = (id: number, newPriority: string) => {
    updateUrgencyMutation.mutate({ inquiryId: id, newPriority: newPriority as Urgency });
  };

  // Paginator 페이지 변경 핸들러
  const onPageChange = (event: PaginatorPageChangeEvent) => {
    setFirst(event.first);
    setRows(event.rows);
  };

  // 4. 달력의 onChange 이벤트를 처리하는 새로운 핸들러입니다.
  const handleDateChange = (e: any) => {
    const dates = e.value;
    setCalendarValue(dates);

    if (Array.isArray(dates) && dates.length === 2 && dates[1] !== null) {
      setDateRangeFilter(dates);
    }
  };

  // 5. 달력의 'x' (초기화) 버튼을 클릭했을 때 호출될 핸들러입니다.
  const handleDateClear = () => {
    setCalendarValue(null);
    setDateRangeFilter(null);
  };

  if (isLoading) return <div>데이터를 불러오는 중입니다...</div>;
  if (isError) return <div>오류가 발생했습니다.</div>;

  return (
    <div className="p-4 bg-white rounded-lg shadow-md">
      <div className="flex flex-col md:flex-row border-2 rounded-t-[20px] border-[#C5C5C5] p-6 gap-6">
        <div className="flex flex-col w-full md:w-1/2 lg:w-2/3">
          <div className="flex-grow"></div>
          <div className="flex items-center gap-2">
            <InputText
              placeholder="제목과 내용으로 검색 가능합니다"
              className="w-full border-0 border-b-2 border-gray-300 rounded-none focus:outline-none focus:ring-0 focus:border-black"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            <Button icon="pi pi-search" className="p-button-outlined" />
          </div>
        </div>

        <div className="grid grid-cols-[auto_1fr] items-center gap-x-4 gap-y-4 w-full md:w-1/2 lg:w-1/3">
          {/* Row 1: 작성 날짜 */}
          <label className="font-semibold whitespace-nowrap">작성 날짜</label>
          <div className="flex items-center">
            <Calendar
              value={calendarValue}
              onChange={handleDateChange}
              selectionMode="range"
              dateFormat="yy/mm/dd"
              readOnlyInput
              placeholder="날짜 범위 선택"
              showIcon
              hideOnRangeSelection
              pt={{ 
                root: { className: 'w-full' },
                input: { className: 'w-full' }
              }}
            />
            {calendarValue && (
              <Button 
                icon="pi pi-times" 
                className="p-button-text p-button-sm !ml-1"
                onClick={handleDateClear} 
              />
            )}
          </div>

          {/* Row 2: 카테고리 */}
          <label className="font-semibold whitespace-nowrap">카테고리</label>
          <Dropdown
            value={category}
            options={categoryOptions}
            onChange={(e) => setCategory(e.value)}
            placeholder="전체"
            showClear
            optionLabel="label"
            optionValue="value"
            className="w-full"
          />

          {/* Row 3: 중요도 */}
          <label className="font-semibold whitespace-nowrap">중요도</label>
          <MultiSelect
            value={urgencies}
            options={priorityOptions}
            onChange={(e) => setUrgencies(e.value)}
            placeholder="전체"
            maxSelectedLabels={2}
            className="w-full"
            showClear 
          />

          {/* Row 4: 답변 처리 */}
          <label className="font-semibold whitespace-nowrap">답변 처리</label>
          <Dropdown
            value={status}
            options={statusOptions}
            onChange={(e) => setStatus(e.value)}
            placeholder="전체"
            showClear
            optionLabel="label"
            optionValue="value"
            className="w-full"
          />
        </div>
      </div>

      {/* List */}
      <div className="space-y-4 p-4 border-2 border-t-0 rounded-b-[20px] border-[#C5C5C5]">
        {inquiriesData?.content && inquiriesData.content.map((item) => (
          <div
            key={item.inquiryId}
            className="border rounded-lg p-4 shadow-sm flex flex-col md:flex-row justify-between items-start md:items-center cursor-pointer hover:bg-gray-50"
            onClick={() => onSelectInquiry(item)}
          >
          <div>
              <h3 className="text-lg font-bold">{item.title}</h3>
              <div className="flex items-center gap-4 text-sm text-gray-500 mt-2 flex-wrap">
                <div className='flex gap-2 items-center'> <UserIcon className="h-4 w-4" /> {item.userId}</div>
                <div className='flex gap-2 items-center'> <CalendarDaysIcon className="h-4 w-4" /> {new Date(item.createdAt).toLocaleDateString()}</div>
                <div className='flex gap-2 items-center'> <TagIcon className="h-4 w-4" /> {item.category}</div>
              </div>
            </div>
          <div className="flex items-center gap-2 mt-2 md:mt-0">
              <div onClick={(e) => e.stopPropagation()}> 
                <CustomPriorityDropdown
                  value={item.urgency}
                  onChange={(newValue) => handlePriorityChange(item.inquiryId, newValue)}
                />
              </div>
              <span
                className={`px-2 py-1 text-xs font-semibold rounded-full ${
                  item.status === 'RESOLVED'
                    ? 'bg-green-100 text-green-800'
                    : 'bg-yellow-100 text-yellow-800'
                }`}
              >
                {item.status === 'RESOLVED' ? '답변완료' : '미처리'}
              </span>
            </div>
          </div>
      ))}
      {inquiriesData?.empty && <div className="text-center p-4">표시할 문의가 없습니다.</div>}
    </div>

      {/* PrimeReact Paginator */}
      <div className="mt-6">
        <Paginator
          first={first}
          rows={rows}
          totalRecords={inquiriesData?.totalElements || 0}
          rowsPerPageOptions={[10, 20, 30]}
          onPageChange={onPageChange}
          template="FirstPageLink PrevPageLink PageLinks NextPageLink LastPageLink CurrentPageReport RowsPerPageDropdown"
          currentPageReportTemplate="{first}부터 {last}까지 / 총 {totalRecords}개"
        />
      </div>
    </div>
  );
}