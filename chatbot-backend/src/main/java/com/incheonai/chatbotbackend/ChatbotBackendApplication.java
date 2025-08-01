package com.incheonai.chatbotbackend;

import io.github.cdimascio.dotenv.Dotenv;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;
import org.springframework.data.mongodb.repository.config.EnableMongoRepositories;
import org.springframework.core.env.Environment;

@EnableJpaRepositories(basePackages = "com.incheonai.chatbotbackend.repository.jpa")
@EnableMongoRepositories(basePackages = "com.incheonai.chatbotbackend.repository.mongodb")
@SpringBootApplication
public class ChatbotBackendApplication {

	public static void main(String[] args) {
		// .env 파일 로드를 Spring Boot의 Environment를 확인한 후에 수행하도록 변경
		SpringApplication app = new SpringApplication(ChatbotBackendApplication.class);
		Environment env = app.run(args).getEnvironment();

		// 활성화된 프로필이 없거나 'local-ssl'일 경우에만 .env 파일을 로드
		if (env.getActiveProfiles().length == 0 || "local-ssl".equals(env.getActiveProfiles()[0])) {
			loadEnv();
		}
	}

	private static void loadEnv() {
		Dotenv dotenv = Dotenv.load();

		// 시스템 속성 설정
		System.setProperty("DB_URL", dotenv.get("DB_URL"));
		System.setProperty("DB_USERNAME", dotenv.get("DB_USERNAME"));
		System.setProperty("DB_PASSWORD", dotenv.get("DB_PASSWORD"));
		System.setProperty("MONGO_DB_URI", dotenv.get("MONGO_DB_URI"));
		System.setProperty("REDIS_HOST", dotenv.get("REDIS_HOST", "localhost")); // 기본값 설정
		System.setProperty("REDIS_PORT", dotenv.get("REDIS_PORT", "6379")); // 기본값 설정
		System.setProperty("REDIS_DB", dotenv.get("REDIS_DB", "0")); // 기본값 설정
		System.setProperty("PASSWORD", dotenv.get("PASSWORD"));
		System.setProperty("JWT_SECRET_KEY", dotenv.get("JWT_SECRET_KEY"));
		System.setProperty("GOOGLE_CLIENT_ID", dotenv.get("GOOGLE_CLIENT_ID"));
		System.setProperty("GOOGLE_CLIENT_SECRET", dotenv.get("GOOGLE_CLIENT_SECRET"));
	}

}