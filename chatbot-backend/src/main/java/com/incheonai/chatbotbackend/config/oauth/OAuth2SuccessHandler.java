package com.incheonai.chatbotbackend.config.oauth;

import com.incheonai.chatbotbackend.config.jwt.JwtTokenProvider;
import com.incheonai.chatbotbackend.domain.jpa.User;
import com.incheonai.chatbotbackend.domain.mongodb.ChatSession;
import com.incheonai.chatbotbackend.repository.mongodb.ChatSessionRepository;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.HttpSession;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.security.core.Authentication;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.security.web.authentication.SimpleUrlAuthenticationSuccessHandler;
import org.springframework.stereotype.Component;
import org.springframework.web.util.UriComponentsBuilder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

@Slf4j
@Component
@RequiredArgsConstructor
public class OAuth2SuccessHandler extends SimpleUrlAuthenticationSuccessHandler {

    @Value("${frontend.url}")
    private String frontendUrl;

    private final JwtTokenProvider jwtTokenProvider;
    private final RedisTemplate<String, Object> redisTemplate;
    // ✅ 세션 처리를 위해 ChatSessionRepository를 주입받습니다.
    private final ChatSessionRepository chatSessionRepository;

    @Override
    public void onAuthenticationSuccess(HttpServletRequest request, HttpServletResponse response, Authentication authentication) throws IOException, ServletException {
        // CustomOAuth2UserService에서 반환한 CustomOAuth2User 객체를 가져옵니다.
        // 이 객체 안에는 DB에 저장된 User 정보가 들어있습니다.
        OAuth2User oAuth2User = (OAuth2User) authentication.getPrincipal();
        // User 정보를 직접 가져옵니다. (DB를 다시 조회할 필요 없음)
        User user = ((CustomOAuth2User) oAuth2User).getUser();

        // JWT 토큰 생성
        String token = jwtTokenProvider.createToken(user.getUserId());

        // Redis에 토큰 저장
        redisTemplate.opsForValue().set(
                token,
                user.getUserId(),
                jwtTokenProvider.getTokenValidTime(),
                TimeUnit.MILLISECONDS
        );
        log.info("OAuth2 로그인 성공. Redis에 토큰 저장. Key: {}", token);

        // --- ✅ 세션 마이그레이션 또는 신규 생성 로직 ---
        HttpSession httpSession = request.getSession(false);
        String sessionId = null;

        if (httpSession != null) {
            String anonymousSessionId = (String) httpSession.getAttribute("oauth_anonymous_session_id");
            log.info("Retrieved anonymousSessionId from HTTP session: {}", anonymousSessionId);

            if (anonymousSessionId != null) {
                Optional<ChatSession> sessionOpt = chatSessionRepository.findById(anonymousSessionId);
                if (sessionOpt.isPresent()) {
                    ChatSession session = sessionOpt.get();
                    session.setUserId(String.valueOf(user.getId())); // 익명 세션에 사용자 ID 연결
                    chatSessionRepository.save(session);
                    sessionId = session.getId(); // 마이그레이션된 세션 ID 사용
                    log.info("익명 세션 {}을 사용자 {}에게 마이그레이션했습니다.", sessionId, user.getUserId());
                }
                // 사용 후에는 HTTP 세션에서 속성 제거
                httpSession.removeAttribute("oauth_anonymous_session_id");
            }
        }

        // 마이그레이션할 세션이 없었던 경우, 새로운 세션을 생성합니다.
        if (sessionId == null) {
            ChatSession newSession = ChatSession.builder()
                    .id(UUID.randomUUID().toString())
                    .userId(String.valueOf(user.getId()))
                    .createdAt(LocalDateTime.now())
                    .lastActivatedAt(LocalDateTime.now())
                    .expiresAt(LocalDateTime.now().plusHours(24))
                    .build();
            chatSessionRepository.save(newSession);
            sessionId = newSession.getId();
            log.info("새로운 채팅 세션 {}을 생성했습니다.", sessionId);
        }
        // --- 로직 종료 ---

        // 프론트엔드로 리디렉션할 URL 생성 (토큰과 세션 ID를 모두 포함)
        String targetUrl = UriComponentsBuilder.fromUriString(frontendUrl)
                .queryParam("token", token)
                .queryParam("sessionId", sessionId) // sessionId 파라미터 추가
                .build()
                .encode(StandardCharsets.UTF_8)
                .toUriString();

        getRedirectStrategy().sendRedirect(request, response, targetUrl);
    }
}
