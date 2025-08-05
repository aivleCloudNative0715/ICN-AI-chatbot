package com.incheonai.chatbotbackend.dto;

import com.incheonai.chatbotbackend.domain.jpa.InquiryAnswer;
import java.time.LocalDateTime;

/**
 * 문의 답변 응답 DTO
 *
 * @param answerId  답변 PK
 * @param inquiryId 문의 PK
 * @param adminId   작성 관리자 ID
 * @param content   답변 내용
 * @param createdAt 생성 시각
 * @param updatedAt 수정 시각
 */
public record InquiryAnswerResponseDto(
        Integer answerId,
        Integer inquiryId,
        String  adminId,
        String  content,
        LocalDateTime createdAt,
        LocalDateTime updatedAt
) {
    public static InquiryAnswerResponseDto fromEntity(InquiryAnswer e) {
        return new InquiryAnswerResponseDto(
                e.getAnswerId(),
                e.getInquiryId(),
                e.getAdminId(),
                e.getContent(),
                e.getCreatedAt(),
                e.getUpdatedAt()
        );
    }
}
