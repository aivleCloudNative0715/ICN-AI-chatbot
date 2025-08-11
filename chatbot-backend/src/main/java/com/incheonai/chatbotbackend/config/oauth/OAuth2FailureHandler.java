package com.incheonai.chatbotbackend.config.oauth;

import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.oauth2.core.OAuth2AuthenticationException;
import org.springframework.security.oauth2.core.OAuth2Error;
import org.springframework.security.web.authentication.SimpleUrlAuthenticationFailureHandler;
import org.springframework.stereotype.Component;
import org.springframework.web.util.UriComponentsBuilder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

@Slf4j
@Component
public class OAuth2FailureHandler extends SimpleUrlAuthenticationFailureHandler {

    @Value("${frontend.url}")
    private String frontendUrl;

    @Override
    public void onAuthenticationFailure(HttpServletRequest request, HttpServletResponse response, AuthenticationException exception) throws IOException, ServletException {
        log.error("OAuth2 인증 실패: {}", exception.getMessage());

        String errorMessage = "로그인에 실패했습니다. 다시 시도해주세요.";

        // CustomOAuth2UserService에서 던진 에러인지 확인
        if (exception instanceof OAuth2AuthenticationException) {
            OAuth2Error error = ((OAuth2AuthenticationException) exception).getError();
            // 우리가 정의한 "deleted_account" 에러 코드인지 확인
            if ("deleted_account".equals(error.getErrorCode())) {
                errorMessage = error.getDescription(); // "탈퇴한 계정입니다..." 메시지 사용
            }
        }

        // 프론트엔드로 리디렉션할 URL에 에러 메시지를 쿼리 파라미터로 추가
        String targetUrl = UriComponentsBuilder.fromUriString(frontendUrl)
                .queryParam("error", errorMessage)
                .build()
                .encode(StandardCharsets.UTF_8)
                .toUriString();

        getRedirectStrategy().sendRedirect(request, response, targetUrl);
    }
}
