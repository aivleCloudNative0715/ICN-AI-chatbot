// src/main/java/com/incheonai/chatbotbackend/service/ChatService.java

package com.incheonai.chatbotbackend.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.incheonai.chatbotbackend.domain.mongodb.ChatMessage;
import com.incheonai.chatbotbackend.domain.mongodb.MessageType;
import com.incheonai.chatbotbackend.domain.mongodb.SenderType;
import com.incheonai.chatbotbackend.dto.AiApiDto;
import com.incheonai.chatbotbackend.dto.WebSocketMessageDto;
import com.incheonai.chatbotbackend.dto.WebSocketResponseDto;
import com.incheonai.chatbotbackend.repository.primary.ChatMessageRepository;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.ResponseEntity;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Service;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDateTime;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class ChatService {

    private static final Logger log = LoggerFactory.getLogger(ChatService.class);

    private final ChatMessageRepository chatMessageRepository;
    private final SimpMessagingTemplate messagingTemplate;
    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;

    @Value("${ai-server.url}")
    private String aiServerUrl;

    public void processMessage(WebSocketMessageDto messageDto) {
        String userId = getUserIdFromSecurityContext();
        MessageType messageType = messageDto.getMessageType();

        // messageType에 따라 적절한 핸들러 호출
        switch (messageType) {
            case text, recommendation, flightinfo:
                handleNewMessage(messageDto, userId);
                break;
            case edit:
                handleEditQuestion(messageDto, userId);
                break;
            case again:
                handleRegenerateAnswer(messageDto, userId);
                break;
            default:
                log.warn("Unsupported message type: {}", messageType);
                sendError(messageDto.getSessionId(), messageDto.getMessageId(), "지원하지 않는 메시지 타입입니다.", messageType);
                break;
        }
    }

    private void handleNewMessage(WebSocketMessageDto messageDto, String userId) {
        ChatMessage userMessage = ChatMessage.builder()
                .id(messageDto.getMessageId())
                .sessionId(messageDto.getSessionId())
                .userId(userId)
                .sender(SenderType.user)
                .content(messageDto.getContent())
                .messageType(messageDto.getMessageType())
                .createdAt(LocalDateTime.now())
                .build();
        chatMessageRepository.save(userMessage);

        AiApiDto.GenerateRequest generateRequest = AiApiDto.GenerateRequest.builder()
                .sessionId(userMessage.getSessionId())
                .messageId(userMessage.getId())
                .content(userMessage.getContent())
                .build();
        callGenerateApiAndRespond(generateRequest, userMessage);
    }

    private void handleEditQuestion(WebSocketMessageDto messageDto, String userId) {
        ChatMessage editedUserMessage = ChatMessage.builder()
                .id(messageDto.getMessageId())
                .sessionId(messageDto.getSessionId())
                .userId(userId)
                .parentId(messageDto.getParentId())
                .sender(SenderType.user)
                .content(messageDto.getContent())
                .messageType(MessageType.edit)
                .createdAt(LocalDateTime.now())
                .build();
        chatMessageRepository.save(editedUserMessage);

        AiApiDto.GenerateRequest generateRequest = AiApiDto.GenerateRequest.builder()
                .sessionId(editedUserMessage.getSessionId())
                .messageId(editedUserMessage.getId())
                .parentId(editedUserMessage.getParentId())
                .content(editedUserMessage.getContent())
                .build();

        callGenerateApiAndRespond(generateRequest, editedUserMessage, MessageType.edit);
    }

    private void handleRegenerateAnswer(WebSocketMessageDto messageDto, String userId) {
        ChatMessage parentQuestion = chatMessageRepository.findById(messageDto.getParentId())
                .orElseThrow(() -> new IllegalArgumentException("재생성을 위한 원본 메시지를 찾을 수 없습니다: " + messageDto.getParentId()));

        AiApiDto.GenerateRequest generateRequest = AiApiDto.GenerateRequest.builder()
                .sessionId(parentQuestion.getSessionId())
                .messageId(parentQuestion.getId())
                .parentId(parentQuestion.getId())
                .content(parentQuestion.getContent())
                .build();
        callGenerateApiAndRespond(generateRequest, parentQuestion, MessageType.again);
    }


    /**
     * AI 서버에 답변 생성을 요청하고 클라이언트에 응답하는 핵심 메소드
     */
    private void callGenerateApiAndRespond(AiApiDto.GenerateRequest generateRequest, ChatMessage userMessage, MessageType responseMessageType) {
        String url = aiServerUrl + "/chatbot/generate";
        String originalQuestionId = (generateRequest.getParentId() != null)
                ? generateRequest.getParentId()
                : userMessage.getId();

        try {
            HttpEntity<AiApiDto.GenerateRequest> requestEntity = new HttpEntity<>(generateRequest);
            log.info("Request to AI Server ({}): {}", url, objectMapper.writeValueAsString(generateRequest));
            ResponseEntity<AiApiDto.GenerateResponse> responseEntity =
                    restTemplate.postForEntity(url, requestEntity, AiApiDto.GenerateResponse.class);
            AiApiDto.GenerateResponse aiResponse = responseEntity.getBody();
            log.info("Response from AI Server: {}", objectMapper.writeValueAsString(aiResponse));

            if (aiResponse != null && aiResponse.getAnswer() != null) {
                ChatMessage botMessage = ChatMessage.builder()
                        .id(UUID.randomUUID().toString())
                        .sessionId(userMessage.getSessionId())
                        .parentId(originalQuestionId)
                        .sender(SenderType.chatbot)
                        .content(aiResponse.getAnswer())
                        .messageType(responseMessageType)
                        .createdAt(LocalDateTime.now())
                        .build();
                chatMessageRepository.save(botMessage);

                WebSocketResponseDto responseDto = WebSocketResponseDto.from(botMessage);
                messagingTemplate.convertAndSend("/topic/chat/" + userMessage.getSessionId(), responseDto);
                requestAndSendRecommendations(userMessage);
            }
        } catch (HttpClientErrorException e) {
            log.error("HttpClientErrorException: Status: {}, Body: {}", e.getStatusCode(), e.getResponseBodyAsString(), e);
            sendError(userMessage.getSessionId(), originalQuestionId, "AI 서버 통신 중 오류가 발생했습니다.", responseMessageType);
        } catch (Exception e) {
            log.error("Exception while calling AI server.", e);
            sendError(userMessage.getSessionId(), originalQuestionId, "AI 서버 응답을 가져오는 데 실패했습니다.", responseMessageType);
        }
    }

    private void callGenerateApiAndRespond(AiApiDto.GenerateRequest generateRequest, ChatMessage userMessage) {
        callGenerateApiAndRespond(generateRequest, userMessage, MessageType.text);
    }

    /**
     * AI 서버에 추천 질문을 요청하고 클라이언트에 전송
     */
    private void requestAndSendRecommendations(ChatMessage userMessage) {
        String url = aiServerUrl + "/chatbot/recommend";
        try {
            AiApiDto.RecommendRequest recommendRequest = AiApiDto.RecommendRequest.builder()
                    .messageId(userMessage.getId())
                    .content(userMessage.getContent())
                    .build();

            HttpEntity<AiApiDto.RecommendRequest> requestEntity = new HttpEntity<>(recommendRequest);

            log.info("Request for recommendation to AI Server ({}): {}", url, objectMapper.writeValueAsString(recommendRequest));

            ResponseEntity<AiApiDto.RecommendResponse> responseEntity =
                    restTemplate.postForEntity(url, requestEntity, AiApiDto.RecommendResponse.class);

            AiApiDto.RecommendResponse recommendResponse = responseEntity.getBody();
            log.info("Recommendation response from AI Server: {}", objectMapper.writeValueAsString(recommendResponse));

            if (recommendResponse != null && recommendResponse.getRecommendedQuestions() != null && !recommendResponse.getRecommendedQuestions().isEmpty()) {
                WebSocketResponseDto responseDto = WebSocketResponseDto.builder()
                        .messageId(UUID.randomUUID().toString())
                        .userMessageId(userMessage.getId())
                        .sessionId(userMessage.getSessionId())
                        .sender(SenderType.chatbot)
                        .content(String.join(";", recommendResponse.getRecommendedQuestions()))
                        .messageType(MessageType.recommendation)
                        .createdAt(LocalDateTime.now())
                        .build();
                messagingTemplate.convertAndSend("/topic/chat/" + userMessage.getSessionId(), responseDto);
            }
        } catch(Exception e) {
            log.error("Failed to get recommendations.", e);
        }
    }

    public void sendError(String sessionId, String userMessageId, String errorMessage, MessageType messageType) {
        WebSocketResponseDto errorDto = WebSocketResponseDto.builder()
                .messageId(UUID.randomUUID().toString())
                .userMessageId(userMessageId)
                .sessionId(sessionId)
                .sender(SenderType.chatbot)
                .content(errorMessage)
                .messageType(messageType) // 오류는 일반 텍스트로 처리
                .createdAt(LocalDateTime.now())
                .build();
        messagingTemplate.convertAndSend("/topic/chat/" + sessionId, errorDto);
    }

    private String getUserIdFromSecurityContext() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        if (authentication == null || !authentication.isAuthenticated() || "anonymousUser".equals(authentication.getPrincipal())) {
            return null;
        }
        Object principal = authentication.getPrincipal();
        if (principal instanceof UserDetails) {
            return ((UserDetails) principal).getUsername();
        } else {
            return principal.toString();
        }
    }
}
