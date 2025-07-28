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
import AdminAnswerBoard from './tab/AdminAnswerBoard';

export default function AdminDashboardMain() {
  const router = useRouter();
  const [activeIndex, setActiveIndex] = useState(0);
  const [selectedInquiry, setSelectedInquiry] = useState<any>(null); // 새로운 상태: 선택된 문의

  const handleLogout = () => {
    localStorage.removeItem('jwt_token');
    localStorage.removeItem('user_role');
    alert('관리자 계정에서 로그아웃되었습니다.');
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
  const handleSelectInquiry = (inquiry: any) => {
    setSelectedInquiry(inquiry);
  };

  // AdminAnswerBoard에서 목록으로 돌아갈 때 호출될 콜백 함수
  const handleBackFromAnswer = () => {
    setSelectedInquiry(null);
  };

  // AdminAnswerBoard에서 답변 등록 시 호출될 콜백 함수
  const handleRegisterAnswer = (answerContent: string, newPriority: string) => {
    console.log('Answer registered:', { inquiry: selectedInquiry, answerContent, newPriority });
    // 여기서 실제 백엔드 API 호출 로직을 구현합니다.
    // 예를 들어, API 호출 후 성공하면 selectedInquiry를 null로 설정하여 목록으로 돌아갑니다.
    setSelectedInquiry(null);
  };

  // 탭 콘텐츠 렌더링 함수
  const renderTabContent = () => {
    // selectedInquiry가 있으면 AdminAnswerBoard를 렌더링
    if (selectedInquiry) {
      return (
        <AdminAnswerBoard
          inquiry={selectedInquiry}
          onBack={handleBackFromAnswer}
          onRegister={handleRegisterAnswer}
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
        {!selectedInquiry && ( // 선택된 문의가 없을 때만 탭 메뉴를 보여줍니다.
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