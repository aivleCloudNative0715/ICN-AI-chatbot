package com.incheonai.chatbotbackend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;
import org.springframework.data.mongodb.repository.config.EnableMongoRepositories;

@EnableJpaRepositories(basePackages = "com.incheonai.chatbotbackend.repository.jpa")
@EnableMongoRepositories(basePackages = "com.incheonai.chatbotbackend.repository.mongodb")
@SpringBootApplication
public class ChatbotBackendApplication {
	public static void main(String[] args) {
		SpringApplication.run(ChatbotBackendApplication.class, args);
	}
}