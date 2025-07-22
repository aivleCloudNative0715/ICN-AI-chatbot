package backend.controller;

import backend.domain.ChatMessage;
import backend.service.AiChatService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api")
public class ChatController {
    private final AiChatService aiChat;
    public ChatController(AiChatService aiChat) {
        this.aiChat = aiChat;
    }

    /**
     * 특정 세션의 대화 내역 조회
     * URL 예시: GET /api/chats/{sessionId}
     */
    @GetMapping("/chats/{sessionId}")
    public List<ChatMessage> history(@PathVariable String sessionId) {
        return aiChat.history(sessionId);
    }

    /**
     * 특정 세션으로 메시지 전송 (Echo)
     * URL 예시: POST /api/chat/{sessionId}
     */
    @PostMapping("/chat/{sessionId}")
    public ResponseEntity<String> chat(
            @PathVariable String sessionId,
            @RequestBody String message
    ) {
        return ResponseEntity.ok(aiChat.chat(sessionId, message));
    }

    /**
     * 전체 대화 내역 조회 (옵션)
     * URL 예시: GET /api/chats
     */
    @GetMapping("/chats")
    public List<ChatMessage> allHistory() {
        return aiChat.historyAll();
    }
}
