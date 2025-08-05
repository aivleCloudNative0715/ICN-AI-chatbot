package com.incheonai.chatbotbackend.config.jwt;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.util.StringUtils;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

@Slf4j
@RequiredArgsConstructor
public class JwtAuthenticationFilter extends OncePerRequestFilter {

    private final JwtTokenProvider jwtTokenProvider;
    private final RedisTemplate<String, Object> redisTemplate;

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain) throws ServletException, IOException {
        String token = resolveToken(request);
        log.debug("Request URI: {}, Token: {}", request.getRequestURI(), token);

        if (StringUtils.hasText(token) && jwtTokenProvider.validateToken(token)) {
            // Redis에 해당 토큰(Key)이 존재하는지 확인 (로그인된 유효한 토큰인지)
            if (Boolean.TRUE.equals(redisTemplate.hasKey(token))) {
                Authentication authentication = jwtTokenProvider.getAuthentication(token);
                SecurityContextHolder.getContext().setAuthentication(authentication);
                log.debug("인증 정보 저장 완료: '{}'", authentication.getName());
            } else {
                log.warn("유효한 토큰이지만 Redis에 존재하지 않습니다. (로그아웃 또는 만료된 토큰) Token: {}", token);
            }
        } else {
            log.debug("유효한 토큰이 없습니다.");
        }

        filterChain.doFilter(request, response);
    }

    // Request Header에서 토큰 정보를 꺼내오기
    private String resolveToken(HttpServletRequest request) {
        String bearerToken = request.getHeader("Authorization");
        if (StringUtils.hasText(bearerToken) && bearerToken.startsWith("Bearer ")) {
            return bearerToken.substring(7);
        }
        return null;
    }
}