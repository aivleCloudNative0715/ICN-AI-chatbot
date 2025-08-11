// src/main/java/com/incheonai/chatbotbackend/dto/AiApiDto.java
package com.incheonai.chatbotbackend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

public class AiApiDto {

    // POST /chatbot/generate 요청 DTO
    @Getter
    @Builder
    public static class GenerateRequest {
        private String sessionId;
        private String messageId;
        private String parentId;
        private String content;
    }

    // POST /chatbot/generate 응답 DTO
    @Getter @Setter @NoArgsConstructor
    public static class GenerateResponse {
        private String answer;
    }

    // POST /chatbot/recommend 요청 DTO
    @Getter
    @Builder
    public static class RecommendRequest {
        private String messageId;
        private String content;
        private String userId;
    }

    // POST /chatbot/recommend 응답 DTO
    @Getter @Setter @NoArgsConstructor
    public static class RecommendResponse {
        @JsonProperty("recommended_questions")
        private List<String> recommendedQuestions;
    }
}