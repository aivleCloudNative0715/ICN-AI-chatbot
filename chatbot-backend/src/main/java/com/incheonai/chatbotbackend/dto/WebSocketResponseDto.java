// src/main/java/com/incheonai/chatbotbackend/dto/WebSocketResponseDto.java

package com.incheonai.chatbotbackend.dto;

import com.incheonai.chatbotbackend.domain.mongodb.ChatMessage;
import com.incheonai.chatbotbackend.domain.mongodb.MessageType;
import com.incheonai.chatbotbackend.domain.mongodb.SenderType;
import lombok.Builder;
import lombok.Getter;

import java.time.LocalDateTime;

@Getter
@Builder
public class WebSocketResponseDto {
    private String messageId;
    private String userMessageId; // <-- 이 필드가 핵심입니다!
    private String sessionId;
    private SenderType sender;
    private String content;
    private MessageType messageType;
    private LocalDateTime createdAt;

    /**
     * ChatMessage 엔티티를 WebSocketResponseDto로 변환하는 정적 메소드
     */
    public static WebSocketResponseDto from(ChatMessage chatMessage) {
        return WebSocketResponseDto.builder()
                .messageId(chatMessage.getId())
                .userMessageId(chatMessage.getParentId())
                .sessionId(chatMessage.getSessionId())
                .sender(chatMessage.getSender())
                .content(chatMessage.getContent())
                .messageType(chatMessage.getMessageType())
                .createdAt(chatMessage.getCreatedAt())
                .build();
    }
}