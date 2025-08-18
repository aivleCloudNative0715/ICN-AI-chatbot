package com.incheonai.chatbotbackend.repository.primary;

import com.incheonai.chatbotbackend.domain.mongodb.ChatSession;
import org.springframework.data.mongodb.repository.MongoRepository;

import java.time.LocalDateTime;
import java.util.Optional;

public interface ChatSessionRepository extends MongoRepository<ChatSession, String> {

    /**
     * 사용자 ID를 기준으로 만료되지 않은 가장 최근 세션을 찾습니다.
     * @param userId 사용자 ID
     * @param now 현재 시간
     * @return Optional<ChatSession>
     */
    Optional<ChatSession> findFirstByUserIdAndExpiresAtAfterOrderByCreatedAtDesc(String userId, LocalDateTime now);
}
