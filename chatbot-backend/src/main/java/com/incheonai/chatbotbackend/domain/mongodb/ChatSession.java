package com.incheonai.chatbotbackend.domain.mongodb;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;
import java.time.LocalDateTime;

@Getter
@Setter
@Builder
@Document(collection = "chat_sessions")
public class ChatSession {

    @Id
    private String id; // session_id

    @Field("user_id")
    private String userId;

    @Field("anonymous_id")
    private String anonymousId;

    @Field("created_at")
    private LocalDateTime createdAt;

    @Field("last_activated_at")
    private LocalDateTime lastActivatedAt;

    @Field("migrated_to_user_id")
    private String migratedToUserId;

    @Field("expires_at")
    private LocalDateTime expiresAt;
}