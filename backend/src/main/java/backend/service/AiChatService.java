package backend.service;

import backend.domain.ChatMessage;
import backend.repository.ChatMessageRepository;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.List;

@Service
public class AiChatService {
    private final ChatMessageRepository repo;

    public AiChatService(ChatMessageRepository repo) {
        this.repo = repo;
    }

    // 메시지 저장·응답 예제 (echo)
    public String chat(String sessionId, String userMsg) {
        repo.save(new ChatMessage(sessionId, "user", userMsg, Instant.now()));
        String reply = "Echo: " + userMsg;
        repo.save(new ChatMessage(sessionId, "assistant", reply, Instant.now()));
        return reply;
    }

    // 세션별 대화 이력 조회
    public List<ChatMessage> history(String sessionId) {
        return repo.findBySessionIdOrderByTimestampAsc(sessionId);
    }

    /** 전체 대화 이력 조회 (시간순 오름차순) */
    public List<ChatMessage> historyAll() {
        return repo.findAll(Sort.by("timestamp").ascending());
    }
}