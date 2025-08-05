package com.incheonai.chatbotbackend.dto;

import com.incheonai.chatbotbackend.domain.mongodb.KnowledgeFile;
import java.util.Date;

/**
 * 지식 파일 응답용 DTO
 *
 * @param fileId        파일 PK
 * @param uploadAdminId 업로드한 관리자 ID
 * @param originalName  원본 파일명
 * @param storedName    저장된 파일명
 * @param fileType      MIME 타입
 * @param fileSize      파일 크기 (bytes)
 * @param uploadedAt    업로드 시각
 * @param isActive      활성화 여부
 * @param description   설명
 */
public record KnowledgeFileDto(
        String fileId,
        String uploadAdminId,
        String originalName,
        String storedName,
        String fileType,
        Long fileSize,
        Date uploadedAt,
        Boolean isActive,
        String description
) {
    /** 엔티티 → DTO 변환 */
    public static KnowledgeFileDto fromEntity(KnowledgeFile entity) {
        return new KnowledgeFileDto(
                entity.getId(),                  // 파일 PK
                entity.getUploadAdminId(),      // 업로드 관리자
                entity.getOriginFilename(),     // 원본 파일명 (originFilename)
                entity.getMinioObjectName(),    // 저장된 오브젝트명 (minioObjectName)
                entity.getFileType(),           // MIME 타입
                entity.getFileSize(),           // 파일 크기
                entity.getUploadedAt(),         // 업로드 시각
                entity.isActive(),              // 활성화 여부
                entity.getDescription()         // 설명
        );
    }
}