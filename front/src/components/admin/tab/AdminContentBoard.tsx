// src/components/admin/tab/AdminContentBoard.tsx
'use client';

import React, { useState } from 'react';
import { Calendar } from 'primereact/calendar';
import { Dropdown } from 'primereact/dropdown';
import { Checkbox } from 'primereact/checkbox';
import { InputText } from 'primereact/inputtext';
import { Paginator, PaginatorPageChangeEvent } from 'primereact/paginator'; // Paginator 임포트
import CustomPriorityDropdown from '@/components/CustomPriorityDropdown';
import { CalendarDaysIcon, TagIcon, UserIcon } from '@heroicons/react/24/outline';
import { useAuth } from '@/contexts/AuthContext';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { getAdminInquiries, updateInquiryUrgency } from '@/lib/api';
import { AdminInquiryDto, Urgency } from '@/lib/types';

interface AdminContentBoardProps {
  type: 'dashboard' | 'inquiry' | 'suggestion' | 'pending' | 'completed';
  onSelectInquiry: (inquiry: AdminInquiryDto) => void;
}

export default function AdminContentBoard({ type, onSelectInquiry }: AdminContentBoardProps) {
  const { token } = useAuth();
  const queryClient = useQueryClient();

  // 페이지네이션 상태 관리
  const [first, setFirst] = useState(0); // 현재 페이지의 첫 아이템 인덱스
  const [rows, setRows] = useState(10); // 한 페이지에 보여줄 아이템 수

  // 필터 상태 관리 (기존과 동일)
  const [dateRange, setDateRange] = useState<[Date | null, Date | null]>([null, null]);
  const [category, setCategory] = useState('전체');
  const [priority, setPriority] = useState<string[]>([]);
  const [status, setStatus] = useState<string[]>([]);
  const [search, setSearch] = useState('');

  const categoryOptions = [
    { label: '전체', value: '전체' },
    { label: '문의사항', value: '문의사항' },
    { label: '건의사항', value: '건의사항' },
  ];
  const priorityOptions = ['HIGH', 'MEDIUM', 'LOW']; // 백엔드 Enum 값과 일치
  const statusOptions = ['PENDING', 'RESOLVED']; // 백엔드 Enum 값과 일치
  const page = first / rows; // 현재 페이지 번호 계산

  const { data: inquiriesData, isLoading, isError } = useQuery({
    queryKey: ['adminInquiries', page, rows, search, category, priority, status], // rows도 key에 추가
    queryFn: () => {
      if (!token) return;
      return getAdminInquiries(token, page, rows, { // page, rows 전달
        search: search || undefined,
        category: category !== '전체' ? category as any : undefined,
        urgency: priority.length > 0 ? priority[0] as Urgency : undefined,
        status: status.length > 0 ? status[0] as any : undefined,
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

  if (isLoading) return <div>데이터를 불러오는 중입니다...</div>;
  if (isError) return <div>오류가 발생했습니다.</div>;

  return (
    <div className="p-4 bg-white rounded-lg shadow-md">
      <div className="flex flex-col lg:flex-row border-2 rounded-t-[20px] border-[#C5C5C5] p-4 gap-4">
        {/* Search Input */}
        <div className="flex items-end gap-4 lg:w-2/3">
          <InputText
            placeholder="검색은 제목으로만 가능"
            className="w-full border-0 border-b-2 border-gray-300 rounded-none focus:outline-none focus:ring-0 focus:border-black"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <i className="pi pi-search font-bold" />
        </div>

        {/* Filter Area*/}
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
              <CustomPriorityDropdown
                value={item.urgency}
                onChange={(newValue) => handlePriorityChange(item.inquiryId, newValue)}
              />
              <span
                className={`px-2 py-1 text-xs font-semibold rounded-full ${
                  item.status === 'RESOLVED'
                    ? 'bg-green-100 text-green-800'
                    : 'bg-yellow-100 text-yellow-800'
                }`}
              >
                {item.status === 'RESOLVED' ? '답변완료' : '처리중'}
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