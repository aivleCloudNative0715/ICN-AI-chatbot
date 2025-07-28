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

const priorityOptions: PriorityOption[] = [
  { label: '높음', value: '높음' },
  { label: '보통', value: '보통' },
  { label: '낮음', value: '낮음' },
];

const getColorClass = (value: string) => {
  switch (value) {
    case '높음':
      return 'bg-orange-500 text-white';
    case '보통':
      return 'bg-blue-500 text-white';
    case '낮음':
      return 'bg-green-400 text-white';
    default:
      return '';
  }
};

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
      pt={{
        item: {
          className: 'p-0 border-0 hover:bg-transparent',
        },
      }}
      itemTemplate={(option) => (
        <div
          className={`rounded-full px-3 py-1 text-sm font-medium text-center ${getColorClass(
            option.value
          )}`}
        >
          {option.label}
        </div>
      )}
      valueTemplate={(option) =>
        option ? (
          <div
            className={`rounded-full px-3 py-1 text-sm font-medium text-center ${getColorClass(
              option.value
            )}`}
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
