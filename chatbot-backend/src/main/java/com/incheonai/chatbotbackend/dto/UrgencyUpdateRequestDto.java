package com.incheonai.chatbotbackend.dto;

/**
 * 문의 긴급도 수정 요청 DTO
 *
 * @param urgency 새 긴급도 (1~5)
 */
public record UrgencyUpdateRequestDto(
        Integer urgency
) {}
