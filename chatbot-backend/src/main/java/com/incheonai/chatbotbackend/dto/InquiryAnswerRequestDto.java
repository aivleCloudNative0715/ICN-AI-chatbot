package com.incheonai.chatbotbackend.dto;

import com.incheonai.chatbotbackend.domain.jpa.Urgency;

public record InquiryAnswerRequestDto(String adminId, String content, Urgency urgency) {}