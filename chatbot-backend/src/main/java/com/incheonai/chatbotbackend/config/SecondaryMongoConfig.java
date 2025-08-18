package com.incheonai.chatbotbackend.config;

import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.mongo.MongoProperties;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.mongodb.MongoDatabaseFactory;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.SimpleMongoClientDatabaseFactory;
import org.springframework.data.mongodb.repository.config.EnableMongoRepositories;

@Configuration
@EnableMongoRepositories(
        basePackages = "com.incheonai.chatbotbackend.repository.secondary",
        mongoTemplateRef = "secondaryMongoTemplate" // 이 이름과 아래 Bean 이름이 일치해야 함
)
public class SecondaryMongoConfig {

    // 1. "mongodb.secondary" 프로퍼티를 읽어 MongoProperties Bean을 생성
    @Bean
    @ConfigurationProperties(prefix = "mongodb.secondary")
    public MongoProperties secondaryMongoProperties() {
        return new MongoProperties();
    }

    // 2. 위에서 만든 MongoProperties Bean을 파라미터로 주입받아 Factory Bean 생성
    @Bean
    public MongoDatabaseFactory secondaryMongoDatabaseFactory(
            @Qualifier("secondaryMongoProperties") MongoProperties mongoProperties) {
        return new SimpleMongoClientDatabaseFactory(mongoProperties.getUri());
    }

    // 3. 위에서 만든 Factory Bean을 파라미터로 주입받아 Template Bean 생성
    @Bean
    public MongoTemplate secondaryMongoTemplate(
            @Qualifier("secondaryMongoDatabaseFactory") MongoDatabaseFactory mongoDatabaseFactory) {
        return new MongoTemplate(mongoDatabaseFactory);
    }
}