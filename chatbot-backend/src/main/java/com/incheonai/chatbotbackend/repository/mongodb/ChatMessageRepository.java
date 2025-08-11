package com.incheonai.chatbotbackend.repository.mongodb;

import com.incheonai.chatbotbackend.domain.mongodb.ChatMessage;
import com.incheonai.chatbotbackend.domain.mongodb.ChatSession;
import org.springframework.data.mongodb.repository.MongoRepository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

public interface ChatMessageRepository extends MongoRepository<ChatMessage, String> {
    /**
     * 세션 ID를 기준으로 모든 채팅 메시지를 생성 시간(오름차순)으로 정렬하여 조회합니다.
     * @param sessionId 채팅 세션의 ID
     * @return 채팅 메시지 목록
     */
    List<ChatMessage> findBySessionIdOrderByCreatedAtAsc(String sessionId);
}
