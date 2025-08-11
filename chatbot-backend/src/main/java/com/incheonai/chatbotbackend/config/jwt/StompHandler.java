// src/main/java/com/incheonai/chatbotbackend/config/jwt/StompHandler.java

package com.incheonai.chatbotbackend.config.jwt;

import lombok.RequiredArgsConstructor;
import org.springframework.messaging.Message;
import org.springframework.messaging.MessageChannel;
import org.springframework.messaging.simp.stomp.StompCommand;
import org.springframework.messaging.simp.stomp.StompHeaderAccessor;
import org.springframework.messaging.support.ChannelInterceptor;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Component;

@RequiredArgsConstructor
@Component
public class StompHandler implements ChannelInterceptor {

    private final JwtTokenProvider jwtTokenProvider;

    @Override
    public Message<?> preSend(Message<?> message, MessageChannel channel) {
        StompHeaderAccessor accessor = StompHeaderAccessor.wrap(message);

        // STOMP CONNECT 요청일 때 JWT 토큰 검증
        if (StompCommand.CONNECT == accessor.getCommand()) {
            String jwtToken = accessor.getFirstNativeHeader("Authorization");

            // 헤더에서 "Bearer " 접두사 제거
            if (jwtToken != null && jwtToken.startsWith("Bearer ")) {
                jwtToken = jwtToken.substring(7);
            }

            // 토큰 유효성 검사
            if (jwtToken != null && jwtTokenProvider.validateToken(jwtToken)) {
                // 토큰이 유효하면 Authentication 객체를 가져와서 SecurityContext에 저장
                // 여기서는 간단히 accessor에 사용자 정보를 저장하는 방식을 사용할 수 있습니다.
                Authentication authentication = jwtTokenProvider.getAuthentication(jwtToken);
                accessor.setUser(authentication);
            }
        }
        return message;
    }
}