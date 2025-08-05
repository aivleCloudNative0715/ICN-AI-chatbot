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
        Integer urgency,
        String status,
        LocalDateTime createdAt
) {
    public static InquiryDetailDto fromEntity(Inquiry in) {
        return new InquiryDetailDto(
                in.getInquiryId(),
                in.getUserId(),
                in.getTitle(),
                in.getContent(),
                in.getCategory(),
                in.getUrgency(),
                in.getStatus().name(),
                in.getCreatedAt()
        );
    }
}