package com.incheonai.chatbotbackend.controller;

import com.incheonai.chatbotbackend.dto.CheckIdRequestDto;
import com.incheonai.chatbotbackend.dto.CheckIdResponseDto;
import com.incheonai.chatbotbackend.dto.LoginRequestDto;
import com.incheonai.chatbotbackend.service.AuthService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.HttpSession;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.util.Map;

@Slf4j
@RestController
@RequiredArgsConstructor
@RequestMapping("/api/auth")
public class AuthController {

    private final AuthService authService;

    @PostMapping("/check-id")
    public ResponseEntity<CheckIdResponseDto> checkId(@Valid @RequestBody CheckIdRequestDto requestDto) {
        boolean isAvailable = authService.checkIdDuplication(requestDto.userId());
        return ResponseEntity.ok(new CheckIdResponseDto(isAvailable));
    }

    @PostMapping("/login")
    public ResponseEntity<Object> login(@Valid @RequestBody LoginRequestDto requestDto) {
        Object loginResponse = authService.login(requestDto);
        return ResponseEntity.ok(loginResponse);
    }

    /**
     * OAuth2 인증 흐름을 시작하는 새로운 엔드포인트
     * 이 엔드포인트가 세션 저장과 리디렉션을 모두 처리합니다.
     */
    @GetMapping("/oauth2/start")
    public void startOAuth2(HttpServletRequest request, HttpServletResponse response,
                            @RequestParam(name = "anonymousSessionId", required = false) String anonymousSessionId) throws IOException {

        if (anonymousSessionId != null && !anonymousSessionId.isEmpty()) {
            // 세션을 가져오거나, 없으면 새로 생성합니다.
            HttpSession httpSession = request.getSession(true);
            httpSession.setAttribute("oauth_anonymous_session_id", anonymousSessionId);
            log.info("익명 세션 ID [{}]를 HttpSession [{}]에 저장했습니다.", anonymousSessionId, httpSession.getId());
        }

        // Spring Security가 처리할 실제 Google 인증 URL로 서버에서 직접 리디렉션합니다.
        // 이 경로는 SecurityConfig에 설정된 baseUri와 일치해야 합니다.
        response.sendRedirect("/api/oauth2/authorization/google");
    }
}
