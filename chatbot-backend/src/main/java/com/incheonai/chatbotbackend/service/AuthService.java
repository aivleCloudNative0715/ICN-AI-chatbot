package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.repository.jpa.AdminRepository;
import com.incheonai.chatbotbackend.repository.jpa.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;
import java.util.concurrent.TimeUnit;

@Service
@RequiredArgsConstructor
public class AuthService {

    private final UserRepository userRepository;
    private final AdminRepository adminRepository;
    private final RedisTemplate<String, Object> redisTemplate;

    public boolean checkIdDuplication(String userId) {
        // User 테이블 또는 Admin 테이블에 아이디가 이미 존재하는지 확인
        boolean isUserExists = userRepository.findByUserId(userId).isPresent();
        boolean isAdminExists = adminRepository.findByAdminId(userId).isPresent();

        if (isUserExists || isAdminExists) {
            return false; // 중복된 아이디가 존재함
        }

        // Redis에 임시 아이디로 저장되어 있는지 확인
        String redisKey = "temp:userId:" + userId;
        if (Boolean.TRUE.equals(redisTemplate.hasKey(redisKey))) {
            return false; // 다른 사용자가 확인 중인 아이디임
        }

        // 중복이 없으면 1분간 임시 아이디로 Redis에 저장
        redisTemplate.opsForValue().set(redisKey, "true", 1, TimeUnit.MINUTES);

        return true; // 사용 가능한 아이디임
    }
}