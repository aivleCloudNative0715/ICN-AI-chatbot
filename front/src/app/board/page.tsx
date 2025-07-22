// src/app/board/page.tsx
'use client';

import React, { useState, useEffect } from 'react';
import BoardSidebar from '../../components/board/BoardSidebar';
import InquiryList from '../../components/board/InquiryList'; // Assuming you have this for displaying
import { BroadPost, PostCategory, PostFilter } from '../../lib/types'; // Import types

// Dummy Data mimicking API response
const dummyPosts: BroadPost[] = [
  {
    post_id: 'post-1',
    author_id: 'user123',
    title: '문의합니다: 로그인 오류',
    content: '로그인 시 간헐적으로 오류가 발생합니다.',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    category: 'inquiry',
    is_deleted: false,
  },
  {
    post_id: 'post-2',
    author_id: 'user456',
    title: '건의사항: 다크 모드 추가',
    content: '눈의 피로를 줄이기 위해 다크 모드 옵션이 있었으면 좋겠습니다.',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    category: 'suggestion',
    is_deleted: false,
  },
  {
    post_id: 'post-3',
    author_id: 'user123', // Same author as post-1
    title: '문의합니다: 비밀번호 재설정 관련',
    content: '비밀번호 재설정 링크가 메일로 오지 않습니다.',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    category: 'inquiry',
    is_deleted: false,
  },
  {
    post_id: 'post-4',
    author_id: 'user789',
    title: '건의사항: UI 개선',
    content: '버튼 디자인과 폰트 크기 조절이 필요해 보입니다.',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    category: 'suggestion',
    is_deleted: false,
  },
  {
    post_id: 'post-5',
    author_id: 'user456',
    title: '문의합니다: 결제 오류',
    content: '결제가 완료되지 않고 오류 메시지가 뜹니다.',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    category: 'inquiry',
    is_deleted: false,
  },
];

export default function BoardPage() {
  const [currentCategory, setCurrentCategory] = useState<PostCategory>('inquiry');
  const [currentFilter, setCurrentFilter] = useState<PostFilter>('all');
  const [filteredPosts, setFilteredPosts] = useState<BroadPost[]>([]);

  // Assume this is your logged-in user's ID
  const currentUser = { id: 'user123', isLoggedIn: true }; // Example user

  useEffect(() => {

    let postsToDisplay = dummyPosts.filter(post => post.category === currentCategory);

    if (currentFilter === 'my' && currentUser.isLoggedIn) {
      postsToDisplay = postsToDisplay.filter(post => post.author_id === currentUser.id);
    }

    setFilteredPosts(postsToDisplay);
  }, [currentCategory, currentFilter, currentUser.id, currentUser.isLoggedIn]);

  const handleCategorySelect = (category: PostCategory, filter: PostFilter) => {
    setCurrentCategory(category);
    setCurrentFilter(filter);
  };

  return (
    <div className="flex">
      <BoardSidebar
        isLoggedIn={currentUser.isLoggedIn}
        onCategorySelect={handleCategorySelect}
      />
      <div className="flex-1 p-4">
        <h2>{currentCategory === 'inquiry' ? '문의 사항' : '건의 사항'} ({currentFilter === 'all' ? '전체' : '내'})</h2>
        {filteredPosts.length > 0 ? (
          <InquiryList inquiries={filteredPosts.map(post => ({
            id: post.post_id,
            type: post.category === 'inquiry' ? 'Q' : 'S', // Assuming 'S' for suggestion type
            title: post.title,
            content: post.content,
            answer: post.category === 'inquiry' && post.post_id === 'post-1' ? '확인 후 조치하겠습니다.' : '', // Example answer for one inquiry
          }))} />
        ) : (
          <p>게시물이 없습니다.</p>
        )}
      </div>
    </div>
  );
}