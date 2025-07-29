package com.incheonai.chatbotbackend.repository.mongodb;

import com.incheonai.chatbotbackend.domain.mongodb.KnowledgeFile;
import org.springframework.data.mongodb.repository.MongoRepository;

public interface KnowledgeFileRepository extends MongoRepository<KnowledgeFile, String> {
    // MongoRepository<도큐먼트 클래스, ID 타입>
}
