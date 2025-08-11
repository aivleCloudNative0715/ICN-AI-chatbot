package com.incheonai.chatbotbackend.config.oauth;

import com.incheonai.chatbotbackend.config.jwt.JwtTokenProvider;
import com.incheonai.chatbotbackend.domain.jpa.User;
import com.incheonai.chatbotbackend.exception.BusinessException;
import com.incheonai.chatbotbackend.repository.jpa.UserRepository;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.http.HttpStatus;
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

    private final JwtTokenProvider jwtTokenProvider;
    private final UserRepository userRepository;
    private final RedisTemplate<String, Object> redisTemplate; // RedisTemplate 주입


    @Override
    public void onAuthenticationSuccess(HttpServletRequest request, HttpServletResponse response, Authentication authentication) throws IOException, ServletException {
        OAuth2User oAuth2User = (OAuth2User) authentication.getPrincipal();

        String googleId = (String) oAuth2User.getAttributes().get("sub");

        // CustomOAuth2UserService에서 사용자 생성을 보장하므로, 이 시점에는 반드시 사용자가 존재해야 합니다.
        User user = userRepository.findByGoogleId(googleId)
                // 예외 타입을 BusinessException으로 변경
                .orElseThrow(() -> new BusinessException(HttpStatus.INTERNAL_SERVER_ERROR, "OAuth 인증 후 사용자 정보를 찾는 데 실패했습니다."));

        // JWT 토큰 생성
        String token = jwtTokenProvider.createToken(user.getUserId());

        redisTemplate.opsForValue().set(
                token, // Key를 토큰 자체로 사용
                user.getUserId(), // Value를 사용자 ID로 사용
                jwtTokenProvider.getTokenValidTime(),
                TimeUnit.MILLISECONDS
        );
        log.info("OAuth2 로그인 성공. Redis에 토큰 저장. Key: {}", token);

        // 프론트엔드로 리다이렉트할 URL 생성 (토큰을 쿼리 파라미터로 포함)
        String targetUrl = UriComponentsBuilder.fromUriString("http://localhost:3000") // 프론트엔드 주소
                .queryParam("token", token)
                .build()
                .encode(StandardCharsets.UTF_8)
                .toUriString();

        getRedirectStrategy().sendRedirect(request, response, targetUrl);
    }
}