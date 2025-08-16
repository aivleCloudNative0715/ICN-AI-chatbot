'use client';

import { Dropdown, DropdownChangeEvent } from 'primereact/dropdown';

interface PriorityOption {
  label: string;
  value: string;
}

interface CustomPriorityDropdownProps {
  value: string;
  onChange: (value: string) => void;
}

const priorityOptions = [
    { label: '높음', value: 'HIGH', color: 'bg-orange-500 text-white' },
    { label: '보통', value: 'MEDIUM', color: 'bg-blue-500 text-white' },
    { label: '낮음', value: 'LOW', color: 'bg-green-400 text-white' },
  ];

export default function CustomPriorityDropdown({
  value,
  onChange,
}: CustomPriorityDropdownProps) {
  return (
    <Dropdown
      value={value}
      options={priorityOptions}
      onChange={(e: DropdownChangeEvent) => onChange(e.value)}
      optionLabel="label"
      className="text-sm w-[122px] border-gray-300 border"
      panelClassName="!p-2 shadow-lg"
      highlightOnSelect={false} // 선택 시 파란 배경 제거
      appendTo="self"
      pt={{
        item: {
          className: 'p-0 border-0 hover:bg-transparent',
        },
      }}
      itemTemplate={(option) => (
        <div
          className={`rounded-full px-3 py-1 text-sm font-medium text-center ${option.color}`}
        >
          {option.label}
        </div>
      )}
      valueTemplate={(option) =>
        option ? (
          <div
            className={`rounded-full px-3 py-1 text-sm font-medium text-center ${option.color}`}
          >
            {option.label}
          </div>
        ) : (
          <span className="text-sm text-gray-400">선택</span>
        )
      }
    />
  );
}
