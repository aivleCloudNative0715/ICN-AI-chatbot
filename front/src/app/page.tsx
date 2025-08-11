import React, { Suspense } from 'react';
import HomePageClient from './HomePageClient';

// 로딩 중에 보여줄 간단한 UI
function Loading() {
  return <div>Loading...</div>;
}

export default function HomePage() {
  return (
    // Suspense로 동적 클라이언트 컴포넌트를 감싸줍니다.
    <Suspense fallback={<Loading />}>
      <HomePageClient />
    </Suspense>
  );
}
