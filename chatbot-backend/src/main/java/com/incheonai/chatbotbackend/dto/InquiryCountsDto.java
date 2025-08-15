package com.incheonai.chatbotbackend.dto;

import lombok.Builder;

// record를 사용하면 final 필드, 생성자, getter, toString 등을 자동으로 만들어줍니다.
@Builder
public record InquiryCountsDto(
        // 전체 및 상태별
        long total,
        long pending,
        long resolved,

        // 카테고리별
        long inquiry, // 문의
        long suggestion, // 건의

        // 중요도별
        long high,
        long medium,
        long low
) {
}