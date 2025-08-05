package com.incheonai.chatbotbackend.dto;

import com.incheonai.chatbotbackend.domain.jpa.AdminRole;

/**
 * 관리자 생성 요청용 DTO
 *
 * @param adminId   로그인 ID
 * @param password  비밀번호
 * @param adminName 관리자 이름
 * @param role      관리자 권한 (ADMIN, SUPER)
 */
public record AdminAddRequestDto(
        String adminId,
        String password,
        String adminName,
        AdminRole role
) {}
