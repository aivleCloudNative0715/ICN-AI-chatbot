package com.incheonai.chatbotbackend.dto;

import com.incheonai.chatbotbackend.domain.jpa.Urgency;

/**
 * 문의 긴급도 수정 요청 DTO
 *
 * @param urgency (HIGH, MEDIUM, LOW)
 */
public record UrgencyUpdateRequestDto(Urgency urgency) {}
