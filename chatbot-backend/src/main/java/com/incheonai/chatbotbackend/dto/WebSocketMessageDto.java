// 예시: WebSocketMessageDto.java
package com.incheonai.chatbotbackend.dto;

import com.incheonai.chatbotbackend.domain.mongodb.MessageType;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class WebSocketMessageDto {
    private String sessionId;
    private String messageId; // 답변 재생성, 질문 수정 시 필요
    private String content;
    private MessageType messageType;
}