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

	@Bean
	public CommandLineRunner initSuperAdmin(AdminRepository adminRepository, PasswordEncoder passwordEncoder) {
		return args -> {
			// "superadmin" 이라는 아이디를 가진 관리자가 있는지 확인
			if (adminRepository.findByAdminId("superadmin").isEmpty()) {
				// application-local.yml에 정의된 기본 비밀번호 사용
				String rawPassword = "qwer1234!!";
				String encodedPassword = passwordEncoder.encode(rawPassword);

				Admin superAdmin = Admin.builder()
						.adminId("superadmin")
						.password(encodedPassword)
						.adminName("슈퍼관리자")
						.role(AdminRole.SUPER)
						.build();
				adminRepository.save(superAdmin);
				System.out.println("Super admin account created.");
			}
		};
	}
}