package com.incheonai.chatbotbackend;

import com.incheonai.chatbotbackend.domain.jpa.Admin;
import com.incheonai.chatbotbackend.domain.jpa.AdminRole;
import com.incheonai.chatbotbackend.repository.jpa.AdminRepository;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;
import org.springframework.data.mongodb.repository.config.EnableMongoRepositories;
import org.springframework.security.crypto.password.PasswordEncoder;

@EnableJpaRepositories(basePackages = "com.incheonai.chatbotbackend.repository.jpa")
@EnableMongoRepositories(basePackages = "com.incheonai.chatbotbackend.repository.mongodb")
@SpringBootApplication
public class ChatbotBackendApplication {
	public static void main(String[] args) {
		SpringApplication.run(ChatbotBackendApplication.class, args);
	}
}