// src/main/java/com/incheonai/chatbotbackend/service/ChatService.java

package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.domain.mongodb.ChatMessage;
import com.incheonai.chatbotbackend.domain.mongodb.MessageType;
import com.incheonai.chatbotbackend.domain.mongodb.SenderType;
import com.incheonai.chatbotbackend.dto.AiApiDto;
import com.incheonai.chatbotbackend.dto.WebSocketMessageDto;
import com.incheonai.chatbotbackend.repository.mongodb.ChatMessageRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDateTime;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class ChatService {

    private final ChatMessageRepository chatMessageRepository;
    private final SimpMessagingTemplate messagingTemplate;
    private final RestTemplate restTemplate; // RestTemplate 주입

    @Value("${ai-server.url}") // application.yml에서 AI 서버 주소 가져오기
    private String aiServerUrl;

    private String getUserIdFromSecurityContext() {
        Object principal = SecurityContextHolder.getContext().getAuthentication().getPrincipal();
        if (principal instanceof UserDetails) {
            return ((UserDetails) principal).getUsername();
        } else {
            return principal.toString();
        }
    }

    public void processAndRespond(WebSocketMessageDto messageDto) {
        // 1. 사용자 메시지 저장
        ChatMessage userMessage = ChatMessage.builder()
                .id(UUID.randomUUID().toString()) // 메시지 ID 생성
                .sessionId(messageDto.getSessionId())
                .sender(SenderType.user)
                .content(messageDto.getContent())
                .messageType(MessageType.text) // 사용자가 보낸 메시지는 text 타입
                .createdAt(LocalDateTime.now())
                .build();
        chatMessageRepository.save(userMessage);

        String userId = getUserIdFromSecurityContext();

        // 2. AI 서버에 답변 요청
        AiApiDto.GenerateRequest generateRequest = AiApiDto.GenerateRequest.builder()
                .sessionId(userMessage.getSessionId())
                .messageId(userMessage.getId())
                .content(userMessage.getContent())
                .userId(userId)
                .build();

        try {
            AiApiDto.GenerateResponse aiResponse = restTemplate.postForObject(
                    aiServerUrl + "/api/generate",
                    generateRequest,
                    AiApiDto.GenerateResponse.class
            );

            // 3. AI 응답 메시지 저장
            if (aiResponse != null && aiResponse.getAnswer() != null) {
                ChatMessage botMessage = ChatMessage.builder()
                        .id(UUID.randomUUID().toString())
                        .sessionId(messageDto.getSessionId())
                        .parentId(userMessage.getId()) // 어떤 메시지에 대한 답변인지 parentId 설정
                        .sender(SenderType.chatbot)
                        .content(aiResponse.getAnswer())
                        .messageType(MessageType.text) // 일반 답변
                        .createdAt(LocalDateTime.now())
                        .build();
                chatMessageRepository.save(botMessage);

                // 4. 클라이언트에게 챗봇 응답 전송
                messagingTemplate.convertAndSend("/topic/chat/" + messageDto.getSessionId(), botMessage);

                // 5. 이어서 추천 질문 요청
                requestAndSendRecommendations(userMessage);
            }

        } catch (Exception e) {
            // AI 서버 통신 오류 처리
            sendError(messageDto.getSessionId(), "AI 서버 응답을 가져오는 데 실패했습니다.");
        }
    }

    private void requestAndSendRecommendations(ChatMessage userMessage) {
        String userId = getUserIdFromSecurityContext();
        AiApiDto.RecommendRequest recommendRequest = AiApiDto.RecommendRequest.builder()
                .messageId(userMessage.getId())
                .content(userMessage.getContent())
                .userId(userId)
                .build();

        try {
            AiApiDto.RecommendResponse recommendResponse = restTemplate.postForObject(
                    aiServerUrl + "/api/recommend",
                    recommendRequest,
                    AiApiDto.RecommendResponse.class
            );

            if (recommendResponse != null && recommendResponse.getRecommendedQuestions() != null) {
                // 추천 질문을 담은 메시지 생성 (DB 저장은 선택사항)
                ChatMessage recommendationMessage = ChatMessage.builder()
                        .id(UUID.randomUUID().toString())
                        .sessionId(userMessage.getSessionId())
                        .sender(SenderType.chatbot)
                        .content(String.join(";", recommendResponse.getRecommendedQuestions())) // content에 질문 목록 저장
                        .messageType(MessageType.recommendation) // 추천 질문 타입
                        .createdAt(LocalDateTime.now())
                        .build();

                // 클라이언트에게 추천 질문 전송
                messagingTemplate.convertAndSend("/topic/chat/" + userMessage.getSessionId(), recommendationMessage);
            }
        } catch(Exception e) {
            // 추천 질문 요청 실패는 에러 메시지를 보내지 않고 조용히 넘어갈 수 있음
            System.err.println("추천 질문을 가져오는 데 실패했습니다: " + e.getMessage());
        }
    }

    public void regenerateAnswer(WebSocketMessageDto messageDto) {
        // 1. 재생성을 요청한 원본 메시지(사용자 질문 또는 챗봇 답변)를 DB에서 찾음
        ChatMessage originalMessage = chatMessageRepository.findById(messageDto.getMessageId())
                .orElseThrow(() -> new IllegalArgumentException("원본 메시지를 찾을 수 없습니다: " + messageDto.getMessageId()));

        // 2. AI 서버에 '재생성' 요청 (parentId를 포함하여 보냄)
        // 재생성은 보통 챗봇의 답변에 대해 이루어지므로, 원본 메시지의 부모(사용자 질문)를 찾아 content를 보낼 수 있음
        ChatMessage parentQuestion = chatMessageRepository.findById(originalMessage.getParentId())
                .orElse(originalMessage); // 만약 부모가 없으면(첫 질문) 원본 메시지 내용을 사용

        String userId = getUserIdFromSecurityContext();
        AiApiDto.GenerateRequest generateRequest = AiApiDto.GenerateRequest.builder()
                .sessionId(originalMessage.getSessionId())
                .messageId(UUID.randomUUID().toString()) // 새 메시지 ID 생성
                .parentId(parentQuestion.getId()) // 원본 '질문' 메시지를 부모로 설정
                .content(parentQuestion.getContent()) // 원본 '질문' 내용을 다시 보냄
                .userId(userId)
                .build();

        try {
            AiApiDto.GenerateResponse aiResponse = restTemplate.postForObject(
                    aiServerUrl + "/api/generate",
                    generateRequest,
                    AiApiDto.GenerateResponse.class
            );

            if (aiResponse != null && aiResponse.getAnswer() != null) {
                // 3. 재생성된 AI 응답을 새 메시지로 저장
                ChatMessage newBotMessage = ChatMessage.builder()
                        .id(generateRequest.getMessageId()) // 요청 시 생성한 ID 사용
                        .sessionId(originalMessage.getSessionId())
                        .parentId(parentQuestion.getId())
                        .sender(SenderType.chatbot)
                        .content(aiResponse.getAnswer())
                        .messageType(MessageType.again) // '다시' 타입으로 설정
                        .createdAt(LocalDateTime.now())
                        .build();
                chatMessageRepository.save(newBotMessage);

                // 4. 클라이언트에게 전송
                messagingTemplate.convertAndSend("/topic/chat/" + originalMessage.getSessionId(), newBotMessage);
            }
        } catch (Exception e) {
            sendError(originalMessage.getSessionId(), "답변을 재생성하는 데 실패했습니다.");
        }
    }

    public void editQuestion(WebSocketMessageDto messageDto) {
        // 1. 수정 대상이 된 '원본 사용자 질문' 메시지를 찾음
        // 클라이언트가 messageId에 '수정 대상 사용자 질문'의 ID를 보내준다고 가정
        ChatMessage originalUserMessage = chatMessageRepository.findById(messageDto.getMessageId())
                .orElseThrow(() -> new IllegalArgumentException("수정할 질문을 찾을 수 없습니다: " + messageDto.getMessageId()));

        // 2. 수정된 내용을 반영하여 '새로운 사용자 질문' 메시지 저장
        ChatMessage editedUserMessage = ChatMessage.builder()
                .id(UUID.randomUUID().toString())
                .sessionId(originalUserMessage.getSessionId())
                .sender(SenderType.user)
                .content(messageDto.getContent()) // 새로 입력된 수정 내용
                .messageType(MessageType.edit) // '수정' 타입
                .createdAt(LocalDateTime.now())
                .build();
        chatMessageRepository.save(editedUserMessage);

        // 3. AI 서버에 '수정된 질문'으로 답변 요청
        String userId = getUserIdFromSecurityContext();
        AiApiDto.GenerateRequest generateRequest = AiApiDto.GenerateRequest.builder()
                .sessionId(editedUserMessage.getSessionId())
                .messageId(editedUserMessage.getId())
                .content(editedUserMessage.getContent())
                .userId(userId)
                .build();

        try {
            AiApiDto.GenerateResponse aiResponse = restTemplate.postForObject(
                    aiServerUrl + "/api/generate",
                    generateRequest,
                    AiApiDto.GenerateResponse.class
            );

            if (aiResponse != null && aiResponse.getAnswer() != null) {
                // 4. AI 응답을 새 메시지로 저장
                ChatMessage botMessage = ChatMessage.builder()
                        .id(UUID.randomUUID().toString())
                        .sessionId(editedUserMessage.getSessionId())
                        .parentId(editedUserMessage.getId()) // '수정된 사용자 질문'을 부모로 설정
                        .sender(SenderType.chatbot)
                        .content(aiResponse.getAnswer())
                        .messageType(MessageType.text)
                        .createdAt(LocalDateTime.now())
                        .build();
                chatMessageRepository.save(botMessage);

                // 5. 클라이언트에게 전송
                messagingTemplate.convertAndSend("/topic/chat/" + editedUserMessage.getSessionId(), botMessage);
            }
        } catch (Exception e) {
            sendError(editedUserMessage.getSessionId(), "수정된 질문에 대한 답변을 가져오는 데 실패했습니다.");
        }
    }

    public void sendError(String sessionId, String errorMessage) {
        // 에러 정보를 담을 별도 DTO를 만들어 보내는 것이 좋음
        // 여기서는 간단히 텍스트 메시지로 보냄
        ChatMessage errorBotMessage = ChatMessage.builder()
                .id(UUID.randomUUID().toString())
                .sessionId(sessionId)
                .sender(SenderType.chatbot)
                .content(errorMessage)
                .messageType(MessageType.text) // 혹은 별도의 error 타입
                .createdAt(LocalDateTime.now())
                .build();
        messagingTemplate.convertAndSend("/topic/chat/" + sessionId, errorBotMessage);
    }
}