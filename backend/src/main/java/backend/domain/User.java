package backend.domain;

import lombok.Data;
import lombok.NoArgsConstructor;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import java.time.Instant;

@Data
@NoArgsConstructor
@Document("users")
public class User {
    @Id
    private String id;
    private String username;
    private String password;    // 암호화된 비밀번호
    private Instant createdAt;
    private String role = "USER";       //사용자 권환(User, Admin)


    /** 일반 사용자 생성자 */
    public User(String username, String password, Instant createdAt) {
        this(username, password, createdAt, "USER");
    }

    /** 역할 지정 생성자 */
    public User(String username, String password, Instant createdAt, String role) {
        this.username = username;
        this.password = password;
        this.createdAt = createdAt;
        this.role = role;
    }
    // getters/setters 생략
}
