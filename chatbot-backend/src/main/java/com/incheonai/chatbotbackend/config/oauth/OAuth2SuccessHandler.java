package com.incheonai.chatbotbackend.config.oauth;

import com.incheonai.chatbotbackend.config.jwt.JwtTokenProvider;
import com.incheonai.chatbotbackend.dto.oauth.OAuthAttributes;
import com.incheonai.chatbotbackend.repository.jpa.UserRepository;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.Authentication;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.security.web.authentication.SimpleUrlAuthenticationSuccessHandler;
import org.springframework.stereotype.Component;
import org.springframework.web.util.UriComponentsBuilder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

@Component
@RequiredArgsConstructor
public class OAuth2SuccessHandler extends SimpleUrlAuthenticationSuccessHandler {

    private final JwtTokenProvider jwtTokenProvider;
    private final UserRepository userRepository;

    @Override
    public void onAuthenticationSuccess(HttpServletRequest request, HttpServletResponse response, Authentication authentication) throws IOException, ServletException {
        OAuth2User oAuth2User = (OAuth2User) authentication.getPrincipal();

        // 구글 ID로 사용자를 찾음
        String googleId = (String) oAuth2User.getAttributes().get("sub");
        String userId = userRepository.findByGoogleId(googleId)
                .orElseThrow(() -> new IllegalArgumentException("Unexpected user"))
                .getUserId();

        // JWT 토큰 생성
        String token = jwtTokenProvider.createToken(userId);

        // 프론트엔드로 리다이렉트할 URL 생성 (토큰을 쿼리 파라미터로 포함)
        String targetUrl = UriComponentsBuilder.fromUriString("https://localhost:3000/oauth/redirect") // 프론트엔드 리다이렉트 주소
                .queryParam("token", token)
                .build()
                .encode(StandardCharsets.UTF_8)
                .toUriString();

        getRedirectStrategy().sendRedirect(request, response, targetUrl);
    }
}