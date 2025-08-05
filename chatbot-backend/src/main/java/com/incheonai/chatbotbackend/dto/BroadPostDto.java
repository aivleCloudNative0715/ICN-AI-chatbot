package com.incheonai.chatbotbackend.dto;

import com.incheonai.chatbotbackend.domain.jpa.BroadPost;
import java.time.LocalDateTime;

/**
 * 게시글 목록 조회 시 각 아이템
 */
public record BroadPostDto(
        Long postId,
        String authorId,
        String title,
        String category,
        LocalDateTime createdAt,
        LocalDateTime updatedAt
) {
    public static BroadPostDto fromEntity(BroadPost e) {
        return new BroadPostDto(
                e.getPostId(),
                e.getAuthorId(),
                e.getTitle(),
                e.getCategory(),
                e.getCreatedAt(),
                e.getUpdatedAt()
        );
    }
}
