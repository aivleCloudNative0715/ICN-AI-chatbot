// src/components/admin/AdminDashboardMain.tsx
'use client';

import { useRouter } from 'next/navigation';
import { TabMenu } from 'primereact/tabmenu'; // TabMenu 임포트
import { MenuItem } from 'primereact/menuitem'; // MenuItem 타입 임포트
import { useState } from 'react'; // useState 임포트
import AdminHeader from '@/components/admin/AdminHeader';
import DashboardTab from './tab/DashboardTab';
import AdminContentBoard from './tab/AdminContentBoard';
import AdminManagePage from './tab/AdminManagePage';

export default function AdminDashboardMain() {
  const router = useRouter();
  const [activeIndex, setActiveIndex] = useState(0); // 활성화된 탭 인덱스 상태

  const handleLogout = () => {
    localStorage.removeItem('jwt_token');
    localStorage.removeItem('user_role');
    alert('관리자 계정에서 로그아웃되었습니다.');
    router.push('/'); // 로그아웃 후 홈 페이지로 리다이렉트
  };

  // 탭 메뉴 아이템 정의
  // command 대신 onClick을 사용하여 내부 상태 변경 및 콘텐츠 렌더링
  const items: MenuItem[] = [
  { 
    label: '대시보드', 
    icon: 'pi pi-home', 
    className: activeIndex === 0 ? 'font-bold text-blue-600' : 'text-gray-700', 
    command: () => setActiveIndex(0) 
  },
  // { 
  //   label: '문의 사항', 
  //   icon: 'pi pi-question-circle', 
  //   className: activeIndex === 1 ? 'font-bold text-blue-600' : 'text-gray-700', 
  //   command: () => setActiveIndex(1) 
  // },
  // { 
  //   label: '건의 사항', 
  //   icon: 'pi pi-lightbulb', 
  //   className: activeIndex === 2 ? 'font-bold text-blue-600' : 'text-gray-700', 
  //   command: () => setActiveIndex(2) 
  // },
  // { 
  //   label: '미처리 건', 
  //   icon: 'pi pi-exclamation-circle', 
  //   className: activeIndex === 3 ? 'font-bold text-blue-600' : 'text-gray-700', 
  //   command: () => setActiveIndex(3) 
  // },
  // { 
  //   label: '완료 건', 
  //   icon: 'pi pi-check-circle', 
  //   className: activeIndex === 4 ? 'font-bold text-blue-600' : 'text-gray-700', 
  //   command: () => setActiveIndex(4) 
  // },
  // { 
  //   label: '리포트', 
  //   icon: 'pi pi-chart-bar', 
  //   className: activeIndex === 5 ? 'font-bold text-blue-600' : 'text-gray-700', 
  //   command: () => setActiveIndex(5) 
  // },
  { 
    label: '관리자', 
    icon: 'pi pi-user', 
    className: activeIndex === 6 ? 'font-bold text-blue-600' : 'text-gray-700', 
    command: () => setActiveIndex(6) 
  },
  { 
    label: '파일 업로드', 
    icon: 'pi pi-upload', 
    className: activeIndex === 7 ? 'font-bold text-blue-600' : 'text-gray-700', 
    command: () => setActiveIndex(7) 
  }
];


  // 탭 콘텐츠 렌더링 함수
  const renderTabContent = () => {
    switch (activeIndex) {
      case 0: // 대시보드
        return <DashboardTab />;
      // case 1:
      //   return <AdminContentBoard type="inquiry" />;
      // case 2:
      //   return <AdminContentBoard type="suggestion" />;
      // case 3:
      //   return <AdminContentBoard type="pending" />;
      // case 4:
      //   return <AdminContentBoard type="completed" />;
      // case 5: // 리포트
      //   return (
      //     <div className="p-4 bg-white rounded-lg shadow-md">
      //       <h2 className="text-2xl font-bold mb-6 text-gray-800">리포트</h2>
      //       <p className="text-gray-700">여기는 주차장 혼잡도 예측 리포트 등 각종 리포트를 볼 수 있는 페이지입니다.</p>
      //       {/* 리포트 차트, 데이터 테이블 등 UI 추가 */}
      //     </div>
      //   );
      case 1: // 관리자
        return (
          <AdminManagePage/>
        );
      case 2: // 파일 업로드
        return (
          <div className="p-4 bg-white rounded-lg shadow-md">
            <h2 className="text-2xl font-bold mb-6 text-gray-800">파일 업로드</h2>
            <p className="text-gray-700">여기는 챗봇 학습을 위한 파일을 업로드하고 관리하는 페이지입니다.</p>
            {/* 파일 업로드 폼, 업로드된 파일 목록 등 UI 추가 */}
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="flex flex-col w-full h-full">
      <AdminHeader onLogoutClick={handleLogout} />
      
      <div className="flex-grow p-8 bg-gray-100">
        {/* TabMenu 컴포넌트 */}
        <div className="mb-8 bg-white rounded-lg shadow-md p-2">
          <TabMenu model={items} activeIndex={activeIndex} onTabChange={(e) => setActiveIndex(e.index)} />
        </div>

        {/* 탭 콘텐츠 렌더링 */}
        {renderTabContent()}
      </div>
    </div>
  );
}
