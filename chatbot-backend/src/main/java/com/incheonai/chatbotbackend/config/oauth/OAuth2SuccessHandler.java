package com.incheonai.chatbotbackend.config.oauth;

import com.incheonai.chatbotbackend.config.jwt.JwtTokenProvider;
import com.incheonai.chatbotbackend.domain.jpa.User;
// ✨ AuthService를 임포트합니다.
import com.incheonai.chatbotbackend.service.AuthService;
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
import java.util.concurrent.TimeUnit;

@Slf4j
@Component
@RequiredArgsConstructor
public class OAuth2SuccessHandler extends SimpleUrlAuthenticationSuccessHandler {

    @Value("${frontend.url}")
    private String frontendUrl;

    private final JwtTokenProvider jwtTokenProvider;
    private final RedisTemplate<String, Object> redisTemplate;
    private final AuthService authService;

    @Override
    public void onAuthenticationSuccess(HttpServletRequest request, HttpServletResponse response, Authentication authentication) throws IOException, ServletException {
        // 1. 인증된 사용자 정보 가져오기
        OAuth2User oAuth2User = (OAuth2User) authentication.getPrincipal();
        User user = ((CustomOAuth2User) oAuth2User).getUser(); // DB에 저장된 User 엔티티

        // 2. HttpSession에서 비회원 세션 ID 가져오기
        String anonymousSessionId = null;
        HttpSession httpSession = request.getSession(false);
        if (httpSession != null) {
            Object sessionAttribute = httpSession.getAttribute("oauth_anonymous_session_id");
            if (sessionAttribute instanceof String) {
                anonymousSessionId = (String) sessionAttribute;
                log.info("OAuth 성공 후 HttpSession에서 익명 세션 ID [{}]를 찾았습니다.", anonymousSessionId);
                // 사용 후에는 반드시 세션에서 제거하여 불필요한 재사용을 막습니다.
                httpSession.removeAttribute("oauth_anonymous_session_id");
            }
        }

        // 3.AuthService의 통합 세션 관리 메서드를 호출하여 최종 세션 ID 결정
        //    이 메서드 안에 모든 조건부 로직(기록 확인, 마이그레이션, 기존 세션 조회, 신규 생성)이 들어있습니다.
        String finalSessionId = authService.findOrCreateActiveSessionForUser(user, anonymousSessionId);

        // 4. JWT 토큰 생성 및 Redis에 저장
        String token = jwtTokenProvider.createToken(user.getUserId());
        redisTemplate.opsForValue().set(
                token,
                user.getUserId(),
                jwtTokenProvider.getTokenValidTime(),
                TimeUnit.MILLISECONDS
        );
        log.info("OAuth2 로그인 성공. JWT 토큰 생성: {}", token);

        // 5. 프론트엔드로 리디렉션할 최종 URL 생성
        String targetUrl = UriComponentsBuilder.fromUriString(frontendUrl)
                .queryParam("token", token)
                .queryParam("sessionId", finalSessionId) // 결정된 최종 세션 ID를 파라미터로 전달
                .build()
                .encode(StandardCharsets.UTF_8)
                .toUriString();

        getRedirectStrategy().sendRedirect(request, response, targetUrl);
    }
}