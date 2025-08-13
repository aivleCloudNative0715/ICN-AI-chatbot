// src/main/java/com/incheonai/chatbotbackend/dto/AiApiDto.java
package com.incheonai.chatbotbackend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

public class AiApiDto {

    /**
     * POST /api/generate 요청 DTO
     * 자바의 camelCase 필드를 JSON의 snake_case 키에 매핑합니다.
     */
    @Getter
    @Builder
    public static class GenerateRequest {
        @JsonProperty("session_id")
        private String sessionId;

        @JsonProperty("message_id")
        private String messageId;

        @JsonProperty("parent_id")
        private String parentId;

        private String content; // JSON 키와 필드명이 동일하므로 어노테이션 생략 가능
    }

    /**
     * POST /api/generate 응답 DTO
     */
    @Getter @Setter @NoArgsConstructor
    public static class GenerateResponse {
        private String answer; // JSON 키와 필드명이 동일
    }

    /**
     * POST /api/recommend 요청 DTO
     * 자바의 camelCase 필드를 JSON의 snake_case 키에 매핑합니다.
     */
    @Getter
    @Builder
    public static class RecommendRequest {
        @JsonProperty("message_id")
        private String messageId;

        private String content; // JSON 키와 필드명이 동일
    }

    /**
     * POST /api/recommend 응답 DTO
     */
    @Getter @Setter @NoArgsConstructor
    public static class RecommendResponse {
        @JsonProperty("recommend_question")
        private List<String> recommendedQuestions;
    }
}