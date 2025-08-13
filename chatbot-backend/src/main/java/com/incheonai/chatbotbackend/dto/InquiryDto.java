package com.incheonai.chatbotbackend.dto;

import com.incheonai.chatbotbackend.domain.jpa.Inquiry;
import java.time.LocalDateTime;

/**
 * 문의 목록용 DTO
 *
 * @param inquiryId  문의 고유 번호 (PK)
 * @param userId     문의 작성자 아이디
 * @param title      문의 제목
 * @param category   문의 분류
 * @param urgency    긴급도 (1~5)
 * @param status     상태 (OPEN, ANSWERED 등)
 * @param createdAt  생성 시각
 */
public record InquiryDto(
        Integer inquiryId,
        String userId,
        String title,
        String category,
        String  urgency,
        String status,
        LocalDateTime createdAt,
        LocalDateTime updatedAt
) {
    public static InquiryDto fromEntity(Inquiry e) {
        return new InquiryDto(
                e.getInquiryId(),
                e.getUserId(),
                e.getTitle(),
                e.getCategory() != null ? e.getCategory().name() : null,
                e.getUrgency() != null ? e.getUrgency().name() : null,
                e.getStatus() != null ? e.getStatus().name() : null,
                e.getCreatedAt(),
                e.getUpdatedAt()
        );
    }
}