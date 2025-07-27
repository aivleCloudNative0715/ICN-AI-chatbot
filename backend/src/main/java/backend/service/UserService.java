package backend.service;

import backend.domain.User;
import backend.repository.UserRepository;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.List;

@Service
public class UserService {
    private final UserRepository repo;
    private final PasswordEncoder encoder;

    public UserService(UserRepository repo, PasswordEncoder encoder) {
        this.repo = repo;
        this.encoder = encoder;
    }

    /** 일반 회원가입 (USER 역할) */
    public User register(String username, String rawPassword) {
        return saveWithRole(username, rawPassword, "USER");
    }

    /** 로그인 검증 (USER/ADMIN 공용) */
    public User login(String username, String rawPassword) {
        User user = repo.findByUsername(username)
                .orElseThrow(() -> new IllegalArgumentException("사용자명이 존재하지 않습니다."));
        if (!encoder.matches(rawPassword, user.getPassword())) {
            throw new IllegalArgumentException("비밀번호가 일치하지 않습니다.");
        }
        return user;
    }

    /** 관리자 전용 로그인 */
    public User loginAdmin(String username, String rawPassword) {
        User user = login(username, rawPassword);
        if (!"ADMIN".equals(user.getRole())) {
            throw new IllegalArgumentException("관리자 권한이 없습니다.");
        }
        return user;
    }

    /** 전체 사용자 리스트 반환 */
    public List<User> findAll() {
        return repo.findAll();
    }

    /**
     * 애플리케이션 시작 시 내부에서만 호출해
     * ADMIN 계정을 생성하도록 할 메서드
     */
    public User initAdmin(String username, String rawPassword) {
        return saveWithRole(username, rawPassword, "ADMIN");
    }

    /** 역할(role)을 지정하여 회원 저장하는 공통 로직 */
    private User saveWithRole(String username, String rawPassword, String role) {
        if (repo.findByUsername(username).isPresent()) {
            throw new IllegalArgumentException("이미 존재하는 사용자명입니다.");
        }
        String hashed = encoder.encode(rawPassword);
        User user = new User(username, hashed, Instant.now());
        user.setRole(role);
        return repo.save(user);
    }
}
