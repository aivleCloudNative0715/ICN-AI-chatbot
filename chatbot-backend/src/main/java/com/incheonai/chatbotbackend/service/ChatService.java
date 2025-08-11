// src/main/java/com/incheonai/chatbotbackend/service/ChatService.java

package com.incheonai.chatbotbackend.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.incheonai.chatbotbackend.domain.mongodb.ChatMessage;
import com.incheonai.chatbotbackend.domain.mongodb.MessageType;
import com.incheonai.chatbotbackend.domain.mongodb.SenderType;
import com.incheonai.chatbotbackend.dto.AiApiDto;
import com.incheonai.chatbotbackend.dto.WebSocketMessageDto;
import com.incheonai.chatbotbackend.dto.WebSocketResponseDto;
import com.incheonai.chatbotbackend.repository.mongodb.ChatMessageRepository;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
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

    /**
     * 모든 WebSocket 메시지를 처리하는 메인 진입점
     * @param messageDto 클라이언트로부터 받은 메시지 DTO
     */
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
                sendError(messageDto.getSessionId(), "지원하지 않는 메시지 타입입니다.");
                break;
        }
    }

    /**
     * 새 질문, 추천 답변 선택, 편명 입력을 처리 (parentId가 없는 경우)
     */
    private void handleNewMessage(WebSocketMessageDto messageDto, String userId) {
        // 1. 사용자 메시지 저장
        ChatMessage userMessage = ChatMessage.builder()
                .id(UUID.randomUUID().toString())
                .sessionId(messageDto.getSessionId())
                .userId(userId)
                .sender(SenderType.user)
                .content(messageDto.getContent())
                .messageType(messageDto.getMessageType()) // 클라이언트가 보낸 타입 그대로 저장
                .createdAt(LocalDateTime.now())
                .build();
        chatMessageRepository.save(userMessage);

        // 2. AI 서버에 답변 요청 (parentId 없음)
        AiApiDto.GenerateRequest generateRequest = AiApiDto.GenerateRequest.builder()
                .sessionId(userMessage.getSessionId())
                .messageId(userMessage.getId())
                .content(userMessage.getContent())
                // parentId는 null (새 메시지)
                .build();

        callGenerateApiAndRespond(generateRequest, userMessage);
    }

    /**
     * 기존 질문 수정 요청을 처리 (parentId가 있는 경우)
     */
    private void handleEditQuestion(WebSocketMessageDto messageDto, String userId) {
        // 1. 수정된 내용을 반영하여 '새로운 사용자 질문' 메시지 저장
        ChatMessage editedUserMessage = ChatMessage.builder()
                .id(UUID.randomUUID().toString())
                .sessionId(messageDto.getSessionId())
                .userId(userId)
                .parentId(messageDto.getParentId()) // 수정 대상이 된 원본 질문의 ID
                .sender(SenderType.user)
                .content(messageDto.getContent()) // 새로 입력된 수정 내용
                .messageType(MessageType.edit)
                .createdAt(LocalDateTime.now())
                .build();
        chatMessageRepository.save(editedUserMessage);

        // 2. AI 서버에 '수정된 질문'으로 답변 요청
        AiApiDto.GenerateRequest generateRequest = AiApiDto.GenerateRequest.builder()
                .sessionId(editedUserMessage.getSessionId())
                .messageId(editedUserMessage.getId())
                .parentId(editedUserMessage.getParentId()) // 부모 ID 포함
                .content(editedUserMessage.getContent())   // 수정된 내용 포함
                .build();

        callGenerateApiAndRespond(generateRequest, editedUserMessage);
    }

    /**
     * 답변 재생성 요청을 처리 (parentId가 있는 경우)
     */
    private void handleRegenerateAnswer(WebSocketMessageDto messageDto, String userId) {
        // 1. 재생성을 요청한 원본 '사용자 질문' 메시지를 DB에서 찾음
        ChatMessage parentQuestion = chatMessageRepository.findById(messageDto.getParentId())
                .orElseThrow(() -> new IllegalArgumentException("재생성을 위한 원본 메시지를 찾을 수 없습니다: " + messageDto.getParentId()));

        // 2. AI 서버에 '재생성' 요청 (parentId와 원본 content 포함)
        AiApiDto.GenerateRequest generateRequest = AiApiDto.GenerateRequest.builder()
                .sessionId(parentQuestion.getSessionId())
                .messageId(UUID.randomUUID().toString()) // 새 메시지 ID 생성
                .parentId(parentQuestion.getId())        // 원본 '질문' 메시지를 부모로 설정
                .content(parentQuestion.getContent())    // 원본 '질문' 내용을 다시 보냄
                .build();

        // 재생성 요청에 대한 AI의 답변은 일반 'text' 타입이 아닌 'again' 타입으로 저장/전송
        callGenerateApiAndRespond(generateRequest, parentQuestion, MessageType.again);
    }


    // AI 응답 처리 및 전송 로직 수정
    private void callGenerateApiAndRespond(AiApiDto.GenerateRequest generateRequest, ChatMessage userMessage, MessageType responseMessageType) {
        String url = aiServerUrl + "/chatbot/generate";
        try {
            log.info("Request to AI Server ({}): {}", url, objectMapper.writeValueAsString(generateRequest));

            AiApiDto.GenerateResponse aiResponse = restTemplate.postForObject(url, generateRequest, AiApiDto.GenerateResponse.class);

            log.info("Response from AI Server: {}", objectMapper.writeValueAsString(aiResponse));

            if (aiResponse != null && aiResponse.getAnswer() != null) {
                // 3. AI 응답 메시지를 DB에 저장
                ChatMessage botMessage = ChatMessage.builder()
                        .id(UUID.randomUUID().toString())
                        .sessionId(userMessage.getSessionId())
                        .parentId(userMessage.getId())
                        .sender(SenderType.chatbot)
                        .content(aiResponse.getAnswer())
                        .messageType(responseMessageType)
                        .createdAt(LocalDateTime.now())
                        .build();
                chatMessageRepository.save(botMessage);

                // 4. 프론트엔드에 보낼 응답 DTO 생성
                WebSocketResponseDto responseDto = WebSocketResponseDto.builder()
                        .messageId(botMessage.getId())             // 챗봇 답변의 ID
                        .userMessageId(botMessage.getParentId())   // 원본 사용자 질문의 ID
                        .sessionId(botMessage.getSessionId())
                        .sender(botMessage.getSender())
                        .content(botMessage.getContent())
                        .messageType(botMessage.getMessageType())
                        .createdAt(botMessage.getCreatedAt())
                        .build();

                // 5. 생성된 DTO를 클라이언트에게 전송
                messagingTemplate.convertAndSend("/topic/chat/" + userMessage.getSessionId(), responseDto);

                if (responseMessageType == MessageType.text) {
                    requestAndSendRecommendations(userMessage);
                }
            }
        } catch (HttpClientErrorException e) {
            log.error("HttpClientErrorException: Status: {}, Body: {}", e.getStatusCode(), e.getResponseBodyAsString(), e);
            sendError(userMessage.getSessionId(), "AI 서버 통신 중 오류가 발생했습니다.");
        } catch (Exception e) {
            log.error("Exception while calling AI server.", e);
            sendError(userMessage.getSessionId(), "AI 서버 응답을 가져오는 데 실패했습니다.");
        }
    }

    private void callGenerateApiAndRespond(AiApiDto.GenerateRequest generateRequest, ChatMessage userMessage) {
        callGenerateApiAndRespond(generateRequest, userMessage, MessageType.text);
    }

    private void requestAndSendRecommendations(ChatMessage userMessage) {
        AiApiDto.RecommendRequest recommendRequest = AiApiDto.RecommendRequest.builder()
                .messageId(userMessage.getId())
                .content(userMessage.getContent())
                .build();

        String url = aiServerUrl + "/chatbot/recommend";
        try {
            log.info("Request for recommendation to AI Server ({}): {}", url, objectMapper.writeValueAsString(recommendRequest));
            AiApiDto.RecommendResponse recommendResponse = restTemplate.postForObject(url, recommendRequest, AiApiDto.RecommendResponse.class);
            log.info("Recommendation response from AI Server: {}", objectMapper.writeValueAsString(recommendResponse));

            if (recommendResponse != null && recommendResponse.getRecommendedQuestions() != null) {
                // 프론트엔드에 보낼 추천 질문 DTO 생성
                WebSocketResponseDto responseDto = WebSocketResponseDto.builder()
                        .messageId(UUID.randomUUID().toString())
                        .userMessageId(userMessage.getId()) // 어떤 질문에 대한 추천인지 명시
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

    public void sendError(String sessionId, String errorMessage) {
        // 프론트엔드에 보낼 에러 DTO 생성
        WebSocketResponseDto errorDto = WebSocketResponseDto.builder()
                .messageId(UUID.randomUUID().toString())
                // userMessageId는 없으므로 null
                .sessionId(sessionId)
                .sender(SenderType.chatbot)
                .content(errorMessage)
                .messageType(MessageType.text) // 혹은 'error' 타입 추가 가능
                .createdAt(LocalDateTime.now())
                .build();
        messagingTemplate.convertAndSend("/topic/chat/" + sessionId, errorDto);
    }

    /**
     * Spring Security의 SecurityContext에서 사용자 ID를 추출합니다.
     * JWT 토큰 기반 인증을 사용하고 있다면, 토큰 파싱 후 SecurityContext에 저장된
     * Authentication 객체에서 사용자 정보를 가져올 수 있습니다.
     * @return 인증된 사용자의 ID, 비로그인 시 null
     */
    private String getUserIdFromSecurityContext() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();

        // 인증 정보가 없거나, 인증되지 않았거나, 익명 사용자인 경우 null 반환
        if (authentication == null || !authentication.isAuthenticated() || "anonymousUser".equals(authentication.getPrincipal())) {
            return null;
        }

        Object principal = authentication.getPrincipal();
        if (principal instanceof UserDetails) {
            // UserDetails 인터페이스를 구현한 객체일 경우 username을 반환 (일반적인 경우)
            return ((UserDetails) principal).getUsername();
        } else {
            // principal이 단순 문자열 등 다른 형태일 경우 toString() 결과를 반환
            return principal.toString();
        }
    }
}