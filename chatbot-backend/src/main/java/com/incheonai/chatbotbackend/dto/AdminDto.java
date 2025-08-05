package com.incheonai.chatbotbackend.dto;

import com.incheonai.chatbotbackend.domain.jpa.Admin;
import java.time.LocalDateTime;

/**
 * 관리자 정보 응답용 DTO
 *
 * @param id           PK
 * @param adminId      로그인 ID
 * @param adminName    관리자 이름
 * @param role         관리자 권한
 * @param createdAt    생성 시각
 * @param updatedAt    수정 시각
 * @param lastLoginAt  마지막 로그인 시각
 */
public record AdminDto(
        Integer id,
        String adminId,
        String adminName,
        String role,
        LocalDateTime createdAt,
        LocalDateTime updatedAt,
        LocalDateTime lastLoginAt,
        Boolean isActive
) {
    public static AdminDto fromEntity(Admin a) {
        return new AdminDto(
                a.getId(),
                a.getAdminId(),
                a.getAdminName(),
                a.getRole().name(),
                a.getCreatedAt(),
                a.getUpdatedAt(),
                a.getLastLoginAt(),
                a.getDeletedAt() == null
        );
    }
}
