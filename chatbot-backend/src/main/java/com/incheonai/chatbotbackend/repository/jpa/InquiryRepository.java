package com.incheonai.chatbotbackend.repository.jpa;

import com.incheonai.chatbotbackend.domain.jpa.BoardCategory;
import com.incheonai.chatbotbackend.domain.jpa.Inquiry;
import com.incheonai.chatbotbackend.domain.jpa.InquiryStatus;
import com.incheonai.chatbotbackend.domain.jpa.Urgency;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Page;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;

import java.util.Optional;
import java.time.LocalDateTime;

public interface InquiryRepository extends JpaRepository<Inquiry, Integer>, JpaSpecificationExecutor<Inquiry> {
    /** 카테고리별 문의/건의 목록 조회 */
    Page<Inquiry> findByCategory(BoardCategory category, Pageable pageable);

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
    Page<Inquiry> findByUserIdAndCategory(String userId, BoardCategory category, Pageable pageable);
    Page<Inquiry> findByUserIdAndCategoryAndStatus(String userId, BoardCategory category, InquiryStatus status, Pageable pageable);

    /**
     * 기간과 카테고리로 문의 건수 조회
     * @param category 문의/건의
     * @param start 시작일
     * @param end 종료일
     * @return 건수
     */
    long countByCategoryAndCreatedAtBetween(BoardCategory category, LocalDateTime start, LocalDateTime end);

    /**
     * 기간과 중요도로 문의 건수 조회
     * @param urgency 높음/중간/낮음
     * @param start 시작일
     * @param end 종료일
     * @return 건수
     */
    long countByUrgencyAndCreatedAtBetween(Urgency urgency, LocalDateTime start, LocalDateTime end);
}
