package com.incheonai.chatbotbackend.dto;

import jakarta.validation.constraints.NotBlank;

/**
 * 게시글 생성/수정 요청 DTO
 */
public record BroadPostRequestDto(
        @NotBlank String authorId,
        @NotBlank String title,
        @NotBlank String content,
        @NotBlank String category
) {}
