package com.incheonai.chatbotbackend.repository.mongodb;

import com.incheonai.chatbotbackend.domain.mongodb.ChatMessage;
import org.springframework.data.mongodb.repository.MongoRepository;

public interface ChatMessageRepository extends MongoRepository<ChatMessage, String> {
}
