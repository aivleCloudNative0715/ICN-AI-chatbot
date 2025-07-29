package com.incheonai.chatbotbackend;

import io.github.cdimascio.dotenv.Dotenv;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;
import org.springframework.data.mongodb.repository.config.EnableMongoRepositories;

@EnableJpaRepositories(basePackages = "com.incheonai.chatbotbackend.repository.jpa")
@EnableMongoRepositories(basePackages = "com.incheonai.chatbotbackend.repository.mongodb")
@SpringBootApplication
public class ChatbotBackendApplication {

	public static void main(String[] args) {
		// .env 파일을 로드합니다.
		Dotenv dotenv = Dotenv.load();

		// .env 파일의 변수들을 시스템 속성으로 설정합니다.
		// 이 작업은 SpringApplication.run()이 실행되기 전에 이루어져야 합니다.
		System.setProperty("DB_URL", dotenv.get("DB_URL"));
		System.setProperty("DB_USERNAME", dotenv.get("DB_USERNAME"));
		System.setProperty("DB_PASSWORD", dotenv.get("DB_PASSWORD"));
		System.setProperty("MONGO_DB_URI", dotenv.get("MONGO_DB_URI"));
		System.setProperty("JWT_SECRET_KEY", dotenv.get("JWT_SECRET_KEY"));
		System.setProperty("GOOGLE_CLIENT_ID", dotenv.get("GOOGLE_CLIENT_ID"));
		System.setProperty("GOOGLE_CLIENT_SECRET", dotenv.get("GOOGLE_CLIENT_SECRET"));
		// --------------------------

		SpringApplication.run(ChatbotBackendApplication.class, args);
	}

}
