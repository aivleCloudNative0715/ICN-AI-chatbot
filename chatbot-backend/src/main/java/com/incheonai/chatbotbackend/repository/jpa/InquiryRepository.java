package com.incheonai.chatbotbackend.repository.jpa;

import com.incheonai.chatbotbackend.domain.jpa.Inquiry;
import com.incheonai.chatbotbackend.domain.jpa.InquiryStatus;
import org.springframework.data.jpa.repository.JpaRepository;

import java.time.LocalDateTime;

public interface InquiryRepository extends JpaRepository<Inquiry, String> {

    /** 주어진 기간 내 전체 문의 건수 */
    long countByCreatedAtBetween(LocalDateTime start, LocalDateTime end);

    /** 특정 상태와 주어진 기간 내 문의 건수 */
    long countByStatusAndCreatedAtBetween(InquiryStatus status, LocalDateTime start, LocalDateTime end);
}
