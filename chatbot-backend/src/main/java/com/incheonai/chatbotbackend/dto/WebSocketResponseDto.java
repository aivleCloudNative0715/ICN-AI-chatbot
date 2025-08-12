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

    // 이 메시지 자체의 고유 ID (챗봇 답변, 추천 질문 등)
    private String messageId;

    // 이 메시지가 응답하는 원본 사용자 메시지의 ID
    private String userMessageId;

    private String sessionId;

    private SenderType sender;

    private String content;

    private MessageType messageType;

    private LocalDateTime createdAt;

    /**
     * ChatMessage 엔티티를 WebSocketResponseDto로 변환하는 정적 팩토리 메소드
     * @param entity 변환할 ChatMessage 객체
     * @return 변환된 WebSocketResponseDto 객체
     */
    public static WebSocketResponseDto from(ChatMessage entity) {
        return WebSocketResponseDto.builder()
                .messageId(entity.getId())
                .userMessageId(entity.getParentId()) // ChatMessage의 parentId가 userMessageId에 해당
                .sessionId(entity.getSessionId())
                .sender(entity.getSender())
                .content(entity.getContent())
                .messageType(entity.getMessageType())
                .createdAt(entity.getCreatedAt())
                .build();
    }
}