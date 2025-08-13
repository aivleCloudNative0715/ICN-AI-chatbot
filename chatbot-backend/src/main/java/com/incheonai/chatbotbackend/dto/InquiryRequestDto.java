package com.incheonai.chatbotbackend.dto;

import com.incheonai.chatbotbackend.domain.jpa.BoardCategory;
import jakarta.validation.constraints.NotBlank;

/**
 * 문의 등록/수정 요청 DTO
 *
 * @param title    제목
 * @param content  본문
 * @param category 카테고리
 */

public record InquiryRequestDto(

        @NotBlank(message = "제목은 필수입니다.")
        String title,

        @NotBlank(message = "내용은 필수입니다.")
        String content,

        @NotBlank(message = "카테고리는 필수입니다.")
        BoardCategory category
) {}
