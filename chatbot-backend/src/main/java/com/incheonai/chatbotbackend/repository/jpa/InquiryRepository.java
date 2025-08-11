package com.incheonai.chatbotbackend.repository.jpa;

import com.incheonai.chatbotbackend.domain.jpa.Inquiry;
import com.incheonai.chatbotbackend.domain.jpa.InquiryStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Page;
import java.util.Optional;
import java.time.LocalDateTime;

public interface InquiryRepository extends JpaRepository<Inquiry, Integer> {

    /** 주어진 기간 내 전체 문의 건수 */
    long countByCreatedAtBetween(LocalDateTime start, LocalDateTime end);

    /** 특정 상태와 주어진 기간 내 문의 건수 */
    long countByStatusAndCreatedAtBetween(InquiryStatus status, LocalDateTime start, LocalDateTime end);

    /** 특정 유저 ID + 상태에 따른 문의 목록 페이징 */
    Page<Inquiry> findByUserIdAndStatus(String userId, String status, Pageable pageable);
    Page<Inquiry> findByUserId(String userId, Pageable pageable);

    /** 사용자 ID + 문의 ID로 조회 */
    Optional<Inquiry> findByInquiryIdAndUserId(Integer inquiryId, String userId);
}
