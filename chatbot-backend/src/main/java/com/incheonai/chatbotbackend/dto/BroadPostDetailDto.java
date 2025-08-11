package com.incheonai.chatbotbackend.dto;

import com.incheonai.chatbotbackend.domain.jpa.BroadPost;
import java.time.LocalDateTime;

/**
 * 게시글 단일 상세 조회 응답 DTO
 */
public record BroadPostDetailDto(
        Long postId,
        String authorId,
        String title,
        String content,
        String category,
        LocalDateTime createdAt,
        LocalDateTime updatedAt
) {
    public static BroadPostDetailDto fromEntity(BroadPost e) {
        return new BroadPostDetailDto(
                e.getPostId(),
                e.getAuthorId(),
                e.getTitle(),
                e.getContent(),
                e.getCategory(),
                e.getCreatedAt(),
                e.getUpdatedAt()
        );
    }
}
