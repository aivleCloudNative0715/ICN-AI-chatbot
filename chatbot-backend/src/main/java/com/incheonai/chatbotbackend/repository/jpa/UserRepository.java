package com.incheonai.chatbotbackend.repository.jpa;

import com.incheonai.chatbotbackend.domain.jpa.User;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;

public interface UserRepository extends JpaRepository<User, Integer> {
    // JpaRepository<엔티티 클래스, PK 타입>

    // 로컬 로그인 시 사용할 아이디로 사용자 조회
    Optional<User> findByUserId(String userId);

    // 구글 로그인 시 사용할 구글 ID로 사용자 조회
    Optional<User> findByGoogleId(String googleId);
}
