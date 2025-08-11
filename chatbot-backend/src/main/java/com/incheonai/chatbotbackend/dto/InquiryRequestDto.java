package com.incheonai.chatbotbackend.dto;

import jakarta.validation.constraints.NotBlank;

/**
 * 문의 등록/수정 요청 DTO
 *
 * @param title    제목
 * @param content  본문
 * @param category 카테고리
 * @param urgency  긴급도
 */

public record InquiryRequestDto(

        @NotBlank(message = "제목은 필수입니다.")
        String title,

        @NotBlank(message = "내용은 필수입니다.")
        String content,

        @NotBlank(message = "카테고리는 필수입니다.")
        String category,

        @NotBlank(message = "긴급도는 필수입니다.")
        Integer urgency
) {}
