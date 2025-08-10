package com.incheonai.chatbotbackend.repository.jpa;

import com.incheonai.chatbotbackend.domain.jpa.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.Optional;

public interface UserRepository extends JpaRepository<User, String> {
    // @Where("deleted_at IS NULL")의 영향을 받음 (활성 사용자만 조회)
    Optional<User> findByUserId(String userId);

    // @Where("deleted_at IS NULL")의 영향을 받음 (활성 사용자만 조회)
    Optional<User> findByGoogleId(String googleId);

    /**
     * [신규] 탈퇴한 회원을 포함하여 가장 최근의 userId 기록을 조회합니다.
     * @param userId
     * @return
     */
    @Query(value = "SELECT * FROM users WHERE user_id = :userId ORDER BY created_at DESC LIMIT 1", nativeQuery = true)
    Optional<User> findLastByUserIdWithDeleted(@Param("userId") String userId);

    /**
     * [신규] 탈퇴한 회원을 포함하여 가장 최근의 googleId 기록을 조회합니다.
     * @param googleId
     * @return
     */
    @Query(value = "SELECT * FROM users WHERE google_id = :googleId ORDER BY created_at DESC LIMIT 1", nativeQuery = true)
    Optional<User> findLastByGoogleIdWithDeleted(@Param("googleId") String googleId);
}
