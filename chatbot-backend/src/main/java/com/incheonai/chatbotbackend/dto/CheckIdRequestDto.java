package com.incheonai.chatbotbackend.dto;

import jakarta.validation.constraints.NotBlank;

// 아이디 중복 확인 요청 DTO
public record CheckIdRequestDto(
        @NotBlank
        String userId
) {}
