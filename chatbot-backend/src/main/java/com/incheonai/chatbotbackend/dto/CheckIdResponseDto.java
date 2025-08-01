package com.incheonai.chatbotbackend.dto;

// 아이디 중복 확인 응답 DTO
public record CheckIdResponseDto(
        boolean isAvailable
) {}
