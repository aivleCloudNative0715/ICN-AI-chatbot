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

    /** 회원가입 */
    public User register(String username, String rawPassword) {
        if (repo.findByUsername(username).isPresent()) {
            throw new IllegalArgumentException("이미 존재하는 사용자명입니다.");
        }
        String hashed = encoder.encode(rawPassword);
        User user = new User(username, hashed, Instant.now());
        return repo.save(user);
    }

    /** 전체 사용자 리스트 반환 */
    public List<User> findAll() {
        return repo.findAll();
    }
}
