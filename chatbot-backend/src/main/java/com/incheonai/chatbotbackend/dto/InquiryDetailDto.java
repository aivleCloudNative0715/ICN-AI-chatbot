package com.incheonai.chatbotbackend.dto;

import com.incheonai.chatbotbackend.domain.jpa.Inquiry;
import java.time.LocalDateTime;

/**
 * 문의 상세 DTO
 *
 * @param inquiryId  문의 고유 번호
 * @param userId     작성자 아이디
 * @param title      제목
 * @param content    본문 내용
 * @param category   분류
 * @param urgency    긴급도
 * @param status     상태
 * @param createdAt  생성 시각
 */
public record InquiryDetailDto(
        Integer inquiryId,
        String userId,
        String title,
        String content,
        String category,
        String urgency,
        String status,
        String answer,
        String adminId,
        LocalDateTime createdAt,
        LocalDateTime updatedAt
) {
    public static InquiryDetailDto fromEntity(Inquiry e) {
        return new InquiryDetailDto(
                e.getInquiryId(),
                e.getUserId(),
                e.getTitle(),
                e.getContent(),
                e.getCategory() != null ? e.getCategory().name() : null,
                e.getUrgency() != null ? e.getUrgency().name() : null,
                e.getStatus() != null ? e.getStatus().name() : null,
                e.getAnswer(),
                e.getAdminId(),
                e.getCreatedAt(),
                e.getUpdatedAt()
        );
    }
}