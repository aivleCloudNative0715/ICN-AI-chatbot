package com.incheonai.chatbotbackend.repository.primary;

import com.incheonai.chatbotbackend.domain.mongodb.RecommendedQuestion;
import org.springframework.data.mongodb.repository.MongoRepository;

public interface RecommendedQuestionRepository extends MongoRepository<RecommendedQuestion, String> {
}

