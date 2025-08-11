package com.incheonai.chatbotbackend.dto;

import com.incheonai.chatbotbackend.domain.jpa.LoginProvider;

/**
 * 로그인 성공 시 응답으로 반환되는 DTO
 * @param accessToken JWT 액세스 토큰
 * @param id 사용자 고유 ID (PK)
 * @param userId 로컬 로그인 아이디
 * @param googleId 구글 로그인 아이디
 * @param loginProvider 로그인 방식 (LOCAL, GOOGLE)
 */
public record LoginResponseDto(
        String accessToken,
        Integer id,
        String userId,
        String googleId,
        LoginProvider loginProvider,
        String sessionId
) {}