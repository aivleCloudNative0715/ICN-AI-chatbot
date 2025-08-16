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

    /** 사용자 ID + 문의 ID로 조회 */
    Optional<Inquiry> findByInquiryIdAndUserId(Integer inquiryId, String userId);

    /** 상태별 문의 건수 */
    long countByStatus(InquiryStatus status);

    /** 카테고리별 문의 건수 */
    long countByCategory(BoardCategory category);

    /** 중요도별 문의 건수 */
    long countByUrgency(Urgency urgency);
}
