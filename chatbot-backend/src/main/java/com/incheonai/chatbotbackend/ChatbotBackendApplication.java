package com.incheonai.chatbotbackend;

import com.incheonai.chatbotbackend.domain.jpa.Admin;
import com.incheonai.chatbotbackend.domain.jpa.AdminRole;
import com.incheonai.chatbotbackend.repository.jpa.AdminRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;
import org.springframework.security.crypto.password.PasswordEncoder;

@EnableJpaRepositories(basePackages = "com.incheonai.chatbotbackend.repository.jpa")
@SpringBootApplication
public class ChatbotBackendApplication {
	public static void main(String[] args) {
		SpringApplication.run(ChatbotBackendApplication.class, args);
	}

	@Bean
	public CommandLineRunner initSuperAdmin(AdminRepository adminRepository, PasswordEncoder passwordEncoder,
											@Value("${super.admin.id}") String adminId,
											@Value("${super.admin.password}") String rawPassword,
											@Value("${super.admin.name}") String adminName) {
		return args -> {
			// "superadmin" 이라는 아이디를 가진 관리자가 있는지 확인
			if (adminRepository.findByAdminId(adminId).isEmpty()) {
				String encodedPassword = passwordEncoder.encode(rawPassword);

				Admin superAdmin = Admin.builder()
						.adminId(adminId)
						.password(encodedPassword)
						.adminName(adminName)
						.role(AdminRole.SUPER)
						.build();
				adminRepository.save(superAdmin);
				System.out.println("Super admin account created.");
			}
		};
	}
}