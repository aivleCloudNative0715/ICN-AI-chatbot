package backend.domain;

import lombok.Data;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import java.time.Instant;

@Data

@Document("users")
public class User {
    @Id
    private String id;
    private String username;
    private String password;    // 암호화된 비밀번호
    private Instant createdAt;

    public User() {}

    public User(String username, String password, Instant createdAt) {
        this.username = username;
        this.password = password;
        this.createdAt = createdAt;
    }
    // getters/setters 생략
}
