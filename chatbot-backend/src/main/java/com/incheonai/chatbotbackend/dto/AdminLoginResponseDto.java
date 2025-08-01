package com.incheonai.chatbotbackend.dto;

import com.incheonai.chatbotbackend.domain.jpa.AdminRole;

/**
 * 관리자 로그인 성공 시 응답으로 반환되는 DTO
 * @param accessToken JWT 액세스 토큰
 * @param id 관리자 고유 ID (PK)
 * @param adminId 관리자 로그인 아이디
 * @param adminName 관리자 이름
 * @param role 권한 (ADMIN, SUPER)
 */
public record AdminLoginResponseDto (
    String accessToken,
    Integer id,
    String adminId,
    String adminName,
    AdminRole role
){}