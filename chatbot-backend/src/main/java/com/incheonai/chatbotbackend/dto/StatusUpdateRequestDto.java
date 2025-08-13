package com.incheonai.chatbotbackend.dto;

import com.incheonai.chatbotbackend.domain.jpa.InquiryStatus;

/**
 * 문의 상태 수정 요청 DTO
 *
 * @param status 새 상태 (PENDING, RESOLVED)
 */
public record StatusUpdateRequestDto(InquiryStatus status) {}
