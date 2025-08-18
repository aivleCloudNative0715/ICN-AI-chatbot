// src/components/admin/AdminDashboardMain.tsx
'use client';

import { useRouter } from 'next/navigation';
import { TabMenu } from 'primereact/tabmenu';
import { MenuItem } from 'primereact/menuitem';
import { useState } from 'react';
import AdminHeader from '@/components/admin/AdminHeader';
import DashboardTab from './tab/DashboardTab';
import AdminManagePage from './tab/AdminManagePage';
import AdminFileUploadTab from './tab/AdminFileUploadTab';
import { AdminInquiryDto } from '@/lib/types';
import AdminAnswerBoard from './tab/AdminAnswerBoard';
import { useAuth } from '@/contexts/AuthContext';

export default function AdminDashboardMain() {
  const router = useRouter();
  const [activeIndex, setActiveIndex] = useState(0);
  const [selectedInquiry, setSelectedInquiry] = useState<AdminInquiryDto | null>(null);

  const { logout } = useAuth();

  const handleLogout = () => {
    logout();
    router.push('/');
  };

  // 탭 메뉴 아이템 정의 (변경 없음)
  const items: MenuItem[] = [
    {
      label: '대시보드',
      icon: 'pi pi-home',
      className: activeIndex === 0 ? 'font-bold text-blue-600' : 'text-gray-700',
      command: () => {
        setActiveIndex(0);
        setSelectedInquiry(null); // 탭 변경 시 선택된 문의 초기화
      },
    },
    {
      label: '관리자',
      icon: 'pi pi-user',
      className: activeIndex === 1 ? 'font-bold text-blue-600' : 'text-gray-700',
      command: () => {
        setActiveIndex(1);
        setSelectedInquiry(null); // 탭 변경 시 선택된 문의 초기화
      },
    },
    {
      label: '파일 업로드',
      icon: 'pi pi-upload',
      className: activeIndex === 2 ? 'font-bold text-blue-600' : 'text-gray-700',
      command: () => {
        setActiveIndex(2);
        setSelectedInquiry(null); // 탭 변경 시 선택된 문의 초기화
      },
    },
  ];

  // AdminContentBoard에서 문의를 선택했을 때 호출될 콜백 함수
  const handleSelectInquiry = (inquiry: AdminInquiryDto) => {
    setSelectedInquiry(inquiry);
  };

  // AdminAnswerBoard에서 목록으로 돌아갈 때 호출될 콜백 함수
  const handleBackFromAnswer = () => {
    setSelectedInquiry(null);
  };


  // 탭 콘텐츠 렌더링 함수
  const renderTabContent = () => {
    if (selectedInquiry) {
      return (
        <AdminAnswerBoard
          // 4. inquiry prop은 이제 inquiryId만 넘겨주는 것이 더 효율적입니다.
          // AdminAnswerBoard는 어차피 id를 이용해 상세 데이터를 다시 불러옵니다.
          inquiry={{ inquiryId: selectedInquiry.inquiryId }}
          onBack={handleBackFromAnswer}
        />
      );
    }

    // 그렇지 않으면 기존 탭 콘텐츠 렌더링
    switch (activeIndex) {
      case 0: // 대시보드
        // DashboardTab에 onSelectInquiry 콜백 함수 전달
        return <DashboardTab onSelectInquiry={handleSelectInquiry} />;
      case 1: // 관리자
        return <AdminManagePage />;
      case 2: // 파일 업로드
        return <AdminFileUploadTab />;
      default:
        return null;
    }
  };

  return (
    <div className="flex flex-col w-full h-full">
      <AdminHeader onLogoutClick={handleLogout} />

      <div className="flex-grow p-8 bg-gray-100">
        {/* TabMenu 컴포넌트 */}
        {/* selectedInquiry가 있을 때는 TabMenu를 숨기거나 비활성화할 수 있습니다. */}
        {!selectedInquiry && ( // 선택된 문의/건의가 없을 때만 탭 메뉴를 보여줍니다.
          <div className="mb-8 bg-white rounded-lg shadow-md p-2">
            <TabMenu model={items} activeIndex={activeIndex} onTabChange={(e) => setActiveIndex(e.index)} />
          </div>
        )}

        {/* 탭 콘텐츠 렌더링 */}
        {renderTabContent()}
      </div>
    </div>
  );
}