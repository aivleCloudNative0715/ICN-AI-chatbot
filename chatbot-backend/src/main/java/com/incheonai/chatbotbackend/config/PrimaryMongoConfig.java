package com.incheonai.chatbotbackend.config;

import org.springframework.boot.autoconfigure.mongo.MongoProperties;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.data.mongodb.MongoDatabaseFactory;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.SimpleMongoClientDatabaseFactory;
import org.springframework.data.mongodb.repository.config.EnableMongoRepositories;

@Configuration
@EnableMongoRepositories(
        basePackages = "com.incheonai.chatbotbackend.repository.primary", // ✅ 1. 경로 수정
        mongoTemplateRef = "primaryMongoTemplate"
)
public class PrimaryMongoConfig {
    // 내용은 이전과 동일
    @Bean
    @Primary
    @ConfigurationProperties(prefix = "spring.data.mongodb")
    public MongoProperties primaryMongoProperties() {
        return new MongoProperties();
    }

    @Bean
    @Primary
    public MongoDatabaseFactory primaryMongoDatabaseFactory(MongoProperties mongoProperties) {
        return new SimpleMongoClientDatabaseFactory(mongoProperties.getUri());
    }

    @Bean
    @Primary
    public MongoTemplate primaryMongoTemplate(MongoDatabaseFactory mongoDatabaseFactory) {
        return new MongoTemplate(mongoDatabaseFactory);
    }
}