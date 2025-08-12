// src/main/java/com/incheonai/chatbotbackend/controller/ChatHistoryController.java

package com.incheonai.chatbotbackend.controller;

import com.incheonai.chatbotbackend.config.jwt.JwtTokenProvider;
import com.incheonai.chatbotbackend.domain.mongodb.ChatMessage;
import com.incheonai.chatbotbackend.domain.mongodb.ChatSession;
import com.incheonai.chatbotbackend.repository.mongodb.ChatMessageRepository;
import com.incheonai.chatbotbackend.repository.mongodb.ChatSessionRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.*;

@RestController
@RequestMapping("api/chat")
@RequiredArgsConstructor
public class ChatHistoryController {

    private final ChatMessageRepository chatMessageRepository;
    private final ChatSessionRepository chatSessionRepository;
    private final JwtTokenProvider jwtTokenProvider;

    /**
     * 채팅 내역 가져오기
     * GET /chat/history?session_id={uuid}
     */
    @GetMapping("/history")
    public ResponseEntity<List<ChatMessage>> getChatHistory(@RequestParam("session_id") String sessionId) {
        // 세션 유효성 검사 (expires_at 기준 24시간 이내)
        ChatSession session = chatSessionRepository.findById(sessionId)
                .orElse(null);

        if (session == null || (session.getExpiresAt() != null && session.getExpiresAt().isBefore(LocalDateTime.now()))) {
            // 세션이 없거나 만료된 경우 빈 목록 반환
            return ResponseEntity.ok(Collections.emptyList());
        }

        List<ChatMessage> history = chatMessageRepository.findBySessionIdOrderByCreatedAtAsc(sessionId);
        return ResponseEntity.ok(history);
    }

    /**
     * 채팅 내역 초기화
     * POST /chat/history/reset
     */
    @PostMapping("/history/reset")
    public ResponseEntity<Map<String, Object>> resetChatHistory(@RequestBody Map<String, String> payload) {
        String oldSessionId = payload.get("old_session_id");

        // 이전 세션 정보를 조회해서 userId를 확보
        ChatSession oldSession = chatSessionRepository.findById(oldSessionId).orElse(null);
        if (oldSession == null) {
            // 이전 세션이 없는 경우의 예외 처리
            return ResponseEntity.badRequest().body(Map.of("message", "Invalid session ID"));
        }

        // 이전 세션 만료 처리
        ChatSession updatedSession = ChatSession.builder()
                .id(oldSession.getId())
                .userId(oldSession.getUserId())
                .anonymousId(oldSession.getAnonymousId())
                .createdAt(oldSession.getCreatedAt())
                .lastActivatedAt(oldSession.getLastActivatedAt())
                .migratedToUserId(oldSession.getMigratedToUserId())
                .expiresAt(LocalDateTime.now()) // 현재 시간으로 만료시킴
                .build();
        chatSessionRepository.save(updatedSession);

        // 새 세션 생성 시 이전 세션의 사용자 정보를 사용
        ChatSession newSession = ChatSession.builder()
                .id(UUID.randomUUID().toString())
                .userId(oldSession.getUserId()) // userId 정보 유지
                .anonymousId(oldSession.getAnonymousId()) // anonymousId 정보 유지
                .createdAt(LocalDateTime.now())
                .lastActivatedAt(LocalDateTime.now())
                .expiresAt(LocalDateTime.now().plusHours(24)) // 24시간 후 만료
                .build();
        chatSessionRepository.save(newSession);

        Map<String, Object> response = Map.of(
                "message", "Chat history reset successfully",
                "new_session_id", newSession.getId()
        );

        return ResponseEntity.ok(response);
    }

    /**
     * 익명/로그인 사용자 모두를 위한 세션 생성 및 조회 API
     * @return 생성된 새 세션 ID
     */
    @PostMapping("/session")
    public ResponseEntity<Map<String, String>> getOrCreateSession(
            @RequestHeader(value = "Authorization", required = false) String authorizationHeader) {

        // 1. 로그인한 사용자인 경우 (Authorization 헤더가 있음)
        if (authorizationHeader != null && authorizationHeader.startsWith("Bearer ")) {
            String jwtToken = authorizationHeader.substring(7);
            String userId = jwtTokenProvider.getUserId(jwtToken); // JWT에서 userId(subject) 추출

            // 2. 이 사용자의 유효한(24시간 이내) 세션이 있는지 확인
            Optional<ChatSession> existingSession = chatSessionRepository
                    .findFirstByUserIdAndExpiresAtAfterOrderByCreatedAtDesc(userId, LocalDateTime.now());

            if (existingSession.isPresent()) {
                // 3. 유효한 세션이 있으면 해당 세션 ID 반환 (세션 유지)
                return ResponseEntity.ok(Map.of("sessionId", existingSession.get().getId()));
            } else {
                // 4. 유효한 세션이 없으면 새로 생성하여 반환
                ChatSession newSession = createNewSession(userId);
                return ResponseEntity.ok(Map.of("sessionId", newSession.getId()));
            }

        } else { // 5. 익명 사용자인 경우 (Authorization 헤더가 없음)
            ChatSession newSession = createNewSession(null); // userId는 null로 설정
            return ResponseEntity.ok(Map.of("sessionId", newSession.getId()));
        }
    }

    // 세션 생성 로직을 별도 private 메소드로 분리하여 재사용
    private ChatSession createNewSession(String userId) {
        ChatSession newSession = ChatSession.builder()
                .id(UUID.randomUUID().toString())
                .userId(userId) // 로그인 사용자는 userId, 익명은 null
                .createdAt(LocalDateTime.now())
                .lastActivatedAt(LocalDateTime.now())
                .expiresAt(LocalDateTime.now().plusHours(24))
                .build();
        return chatSessionRepository.save(newSession);
    }
}