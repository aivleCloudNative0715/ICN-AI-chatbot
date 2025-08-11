package com.incheonai.chatbotbackend.domain.mongodb;

import lombok.Builder;
import lombok.Getter;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;
import java.time.LocalDateTime;

@Getter
@Builder
@Document(collection = "chat_messages")
public class ChatMessage {

    @Id
    private String id; // message_id

    @Field("user_id")
    private String userId;

    @Field("session_id")
    private String sessionId;

    @Field("parent_id")
    private String parentId;

    @Field("sender")
    private SenderType sender;

    @Field("content")
    private String content;

    @Field("message_type")
    private MessageType messageType;

    @Field("is_analysed")
    private boolean isAnalysed;

    @Field("created_at")
    private LocalDateTime createdAt;
}
