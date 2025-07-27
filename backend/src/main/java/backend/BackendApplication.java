package backend;

import backend.service.UserService;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class BackendApplication {

	public static void main(String[] args) {
		SpringApplication.run(BackendApplication.class, args);
	}

	/**
	 * 애플리케이션 시작 시점에 ADMIN 계정이 없으면 생성해 줍니다.
	 * (이미 존재하면 예외가 발생하므로 try/catch 로 무시)
	 */
	@Bean
	public CommandLineRunner initAdmin(UserService userService) {
		return args -> {
			try {
				userService.initAdmin("admin", "admin123");

			} catch (IllegalArgumentException ignored) {
				// 이미 생성되어 있는 경우
			}
		};
	}
}
