package com.incheonai.chatbotbackend.repository.primary;

import com.incheonai.chatbotbackend.domain.mongodb.ChatMessage;
import org.springframework.data.mongodb.repository.MongoRepository;

import java.util.List;

public interface ChatMessageRepository extends MongoRepository<ChatMessage, String> {
    /**
     * 세션 ID를 기준으로 모든 채팅 메시지를 생성 시간(오름차순)으로 정렬하여 조회합니다.
     * @param sessionId 채팅 세션의 ID
     * @return 채팅 메시지 목록
     */
    List<ChatMessage> findBySessionIdOrderByCreatedAtAsc(String sessionId);

    /**
     * 해당 세션 ID를 가진 문서(채팅 메시지)가 하나라도 존재하는지 확인하는 메서드.
     * count 쿼리를 사용하여 성능에 유리합니다.
     */
    boolean existsBySessionId(String sessionId);
}
