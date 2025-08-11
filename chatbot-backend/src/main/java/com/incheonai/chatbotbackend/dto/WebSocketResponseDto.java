// src/main/java/com/incheonai/chatbotbackend/dto/WebSocketResponseDto.java

package com.incheonai.chatbotbackend.dto;

import com.incheonai.chatbotbackend.domain.mongodb.MessageType;
import com.incheonai.chatbotbackend.domain.mongodb.SenderType;
import lombok.Builder;
import lombok.Getter;

import java.time.LocalDateTime;

@Getter
@Builder
public class WebSocketResponseDto {

    // 이 메시지 자체의 고유 ID (챗봇 답변, 추천 질문 등)
    private String messageId;

    // 이 메시지가 응답하는 원본 사용자 메시지의 ID
    private String userMessageId;

    private String sessionId;

    private SenderType sender;

    private String content;

    private MessageType messageType;

    private LocalDateTime createdAt;
}