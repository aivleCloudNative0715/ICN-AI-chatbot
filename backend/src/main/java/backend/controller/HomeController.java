package backend.controller;

import backend.domain.ChatMessage;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class HomeController {

    @GetMapping("/")
    public String home() {
        return "항공 AI 챗봇 백엔드가 정상 기동 중입니다!";
    }
}
