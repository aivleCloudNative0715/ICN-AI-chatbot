package com.incheonai.chatbotbackend.repository.jpa;

import com.incheonai.chatbotbackend.domain.jpa.Inquiry;
import com.incheonai.chatbotbackend.domain.jpa.InquiryStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Page;
import java.util.Optional;
import java.time.LocalDateTime;

public interface InquiryRepository extends JpaRepository<Inquiry, Integer> {
    // @Where 어노테이션 덕분에 삭제된 게시글은 자동으로 제외됩니다.
    Page<Inquiry> findAll(Pageable pageable);

    /** 주어진 기간 내 전체 문의 건수 */
    long countByCreatedAtBetween(LocalDateTime start, LocalDateTime end);

    /** 특정 상태와 주어진 기간 내 문의 건수 */
    long countByStatusAndCreatedAtBetween(InquiryStatus status, LocalDateTime start, LocalDateTime end);

    /** 특정 유저 ID + 상태에 따른 문의 목록 페이징 */
    Page<Inquiry> findByUserIdAndStatus(String userId, InquiryStatus status, Pageable pageable);
    Page<Inquiry> findByUserId(String userId, Pageable pageable);

    /** 사용자 ID + 문의 ID로 조회 */
    Optional<Inquiry> findByInquiryIdAndUserId(Integer inquiryId, String userId);
}
