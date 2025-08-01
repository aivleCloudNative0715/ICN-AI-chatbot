package com.incheonai.chatbotbackend.domain.mongodb;

import lombok.Builder;
import lombok.Getter;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;
import java.time.LocalDateTime;

@Getter
@Builder
@Document(collection = "recommended_questions")
public class RecommendedQuestion {

    @Id
    private String id;

    @Field("message_id")
    private String messageId;

    @Field("trigger_type")
    private String triggerType;

    @Field("question")
    private String question;

    @Field("source_type")
    private String sourceType;

    @Field("created_at")
    private LocalDateTime createdAt;

    @Field("updated_at")
    private LocalDateTime updatedAt;

    @Field("status")
    private boolean status;
}
