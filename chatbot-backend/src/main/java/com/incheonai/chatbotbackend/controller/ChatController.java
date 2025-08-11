// src/main/java/com/incheonai/chatbotbackend/controller/ChatController.java

package com.incheonai.chatbotbackend.controller;

import com.incheonai.chatbotbackend.dto.WebSocketMessageDto;
import com.incheonai.chatbotbackend.service.ChatService;
import lombok.RequiredArgsConstructor;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.simp.SimpMessageHeaderAccessor;
import org.springframework.stereotype.Controller;

@RequiredArgsConstructor
@Controller
public class ChatController {

    private final ChatService chatService;

    /**
     * 클라이언트가 보내는 모든 메시지를 처리하는 단일 엔드포인트.
     * /app/chat.sendMessage 로 메시지를 발행하면,
     * messageDto의 messageType에 따라 적절한 서비스 로직이 호출됩니다.
     */
    @MessageMapping("/chat.sendMessage")
    public void sendMessage(WebSocketMessageDto messageDto) {
        // ChatService의 processMessage가 모든 메시지 유형을 처리
        chatService.processMessage(messageDto);
    }
}