package com.incheonai.chatbotbackend.repository.mongodb;

import com.incheonai.chatbotbackend.domain.mongodb.ChatSession;
import org.springframework.data.mongodb.repository.MongoRepository;

public interface ChatSessionRepository extends MongoRepository<ChatSession, String> {
}
