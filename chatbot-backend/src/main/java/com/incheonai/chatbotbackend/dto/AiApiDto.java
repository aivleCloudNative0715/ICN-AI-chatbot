// AiApiDto.java (하나의 파일에 inner class로 관리해도 좋습니다)
package com.incheonai.chatbotbackend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

public class AiApiDto {

    // POST /api/generate 요청 DTO
    @Getter
    @Builder
    public static class GenerateRequest {
        private String sessionId;
        private String messageId;
        private String parentId;
        private String userId;
        private String content;
    }

    // POST /api/generate 응답 DTO
    @Getter @Setter @NoArgsConstructor
    public static class GenerateResponse {
        private String answer;
    }

    // POST /api/recommend 요청 DTO
    @Getter
    @Builder
    public static class RecommendRequest {
        private String messageId;
        private String content;
        private String userId;
    }

    // POST /api/recommend 응답 DTO
    @Getter @Setter @NoArgsConstructor
    public static class RecommendResponse {
        @JsonProperty("recommended_questions")
        private List<String> recommendedQuestions;
    }
}