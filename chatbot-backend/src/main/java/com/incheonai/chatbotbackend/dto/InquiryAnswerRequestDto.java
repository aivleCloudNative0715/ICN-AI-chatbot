package com.incheonai.chatbotbackend.dto;

// 답변 등록/수정 시 사용할 DTO
public record InquiryAnswerRequestDto(String adminId, String content) {}
