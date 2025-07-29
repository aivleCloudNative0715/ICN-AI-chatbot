package com.incheonai.chatbotbackend.domain.jpa;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.SQLDelete;
import org.hibernate.annotations.UpdateTimestamp;
import org.hibernate.annotations.Where;

import java.time.LocalDateTime;

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED) // JPA는 기본 생성자를 필요로 합니다. (접근 제어자는 PROTECTED 권장)
@Entity
@Table(name = "users") // 'user'는 데이터베이스 예약어일 수 있으므로 'users'와 같이 복수형을 사용하는 것이 안전합니다.
@SQLDelete(sql = "UPDATE users SET deleted_at = CURRENT_TIMESTAMP WHERE id = ?") // delete 쿼리 발생 시 대신 실행될 SQL
@Where(clause = "deleted_at IS NULL") // SELECT 쿼리 시 항상 deleted_at이 NULL인 데이터만 조회
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY) // 데이터베이스의 auto-increment 설정을 따름
    private Integer id;

    @Column(name = "user_id", unique = true) // 로컬 로그인 사용자의 아이디 (유니크 제약조건)
    private String userId;

    @Column(name = "google_id", unique = true) // 구글 계정의 고유 ID (유니크 제약조건)
    private String googleId;

    @Column
    private String password;

    @Enumerated(EnumType.STRING) // Enum의 이름을 DB에 문자열로 저장 (e.g., "LOCAL", "GOOGLE")
    @Column(name = "login_provider", nullable = false)
    private LoginProvider loginProvider;

    @CreationTimestamp // 엔티티가 처음 저장될 때 자동으로 현재 시간 저장
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    @UpdateTimestamp // 엔티티가 업데이트될 때마다 자동으로 현재 시간 저장
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    @Column(name = "last_login_at")
    private LocalDateTime lastLoginAt;

    @Column(name = "deleted_at")
    private LocalDateTime deletedAt;

    @Builder
    public User(String userId, String googleId, String password, LoginProvider loginProvider) {
        this.userId = userId;
        this.googleId = googleId;
        this.password = password;
        this.loginProvider = loginProvider;
    }

    // 마지막 로그인 시간을 업데이트하는 편의 메서드
    public void updateLastLogin() {
        this.lastLoginAt = LocalDateTime.now();
    }
}
