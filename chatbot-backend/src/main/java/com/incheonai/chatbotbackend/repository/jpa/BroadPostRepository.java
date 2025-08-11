package com.incheonai.chatbotbackend.repository.jpa;

import com.incheonai.chatbotbackend.domain.jpa.BroadPost;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

public interface BroadPostRepository extends JpaRepository<BroadPost, Long> {
    // 삭제되지 않은 글만 페이징 조회
    Page<BroadPost> findAllByIsDeletedFalse(Pageable pageable);

    // 카테고리+제목/내용 검색 + 삭제되지 않은 것
    Page<BroadPost> findByIsDeletedFalseAndCategoryAndTitleContainingOrContentContaining(
            String category, String titleKeyword, String contentKeyword, Pageable pageable);
}
