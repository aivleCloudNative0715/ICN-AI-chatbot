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
     * 클라이언트가 메시지를 보낼 때 (sendMessage)
     * /app/chat.sendMessage 로 메시지를 발행
     */
    @MessageMapping("/chat.sendMessage")
    public void sendMessage(WebSocketMessageDto messageDto, SimpMessageHeaderAccessor headerAccessor) {
        // 현재는 connect 이벤트에 대한 별도 처리가 없으므로 sendMessage에서 세션 ID를 받아 처리
        // 또는 connect 시점에 세션 정보를 저장해두고 활용할 수 있음
        chatService.processAndRespond(messageDto);
    }

    /**
     * 답변 재생성 요청 (regenerateAnswer)
     * /app/chat.regenerateAnswer 로 메시지를 발행
     */
    @MessageMapping("/chat.regenerateAnswer")
    public void regenerateAnswer(WebSocketMessageDto messageDto, SimpMessageHeaderAccessor headerAccessor) {
        chatService.regenerateAnswer(messageDto);
    }

    /**
     * 질문 수정 요청 (editQuestion)
     * /app/chat.editQuestion 로 메시지를 발행
     */
    @MessageMapping("/chat.editQuestion")
    public void editQuestion(WebSocketMessageDto messageDto, SimpMessageHeaderAccessor headerAccessor) {
        chatService.editQuestion(messageDto);
    }
}