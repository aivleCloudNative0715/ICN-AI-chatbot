package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.config.jwt.JwtTokenProvider;
import com.incheonai.chatbotbackend.domain.jpa.Admin;
import com.incheonai.chatbotbackend.domain.jpa.User;
import com.incheonai.chatbotbackend.dto.AdminLoginResponseDto;
import com.incheonai.chatbotbackend.dto.LoginRequestDto;
import com.incheonai.chatbotbackend.dto.LoginResponseDto;
import com.incheonai.chatbotbackend.repository.jpa.AdminRepository;
import com.incheonai.chatbotbackend.repository.jpa.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Optional;
import java.util.concurrent.TimeUnit;

@Service
@RequiredArgsConstructor
public class AuthService {

    private final UserRepository userRepository;
    private final AdminRepository adminRepository;
    private final RedisTemplate<String, Object> redisTemplate;
    private final PasswordEncoder passwordEncoder;
    private final JwtTokenProvider jwtTokenProvider;

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

    @Transactional
    public Object login(LoginRequestDto requestDto) {
        // 1. User 테이블에서 아이디 확인
        Optional<User> userOptional = userRepository.findByUserId(requestDto.userId());
        if (userOptional.isPresent()) {
            User user = userOptional.get();
            // 비밀번호 확인
            if (!passwordEncoder.matches(requestDto.password(), user.getPassword())) {
                throw new IllegalArgumentException("잘못된 비밀번호입니다.");
            }
            // 마지막 로그인 시간 업데이트 및 토큰 생성
            user.updateLastLogin();
            String token = jwtTokenProvider.createToken(user.getUserId());
            // 사용자용 응답 DTO 반환
            return new LoginResponseDto(token, user.getId(), user.getUserId(), user.getGoogleId(), user.getLoginProvider());
        }

        // 2. Admin 테이블에서 아이디 확인
        Optional<Admin> adminOptional = adminRepository.findByAdminId(requestDto.userId());
        if (adminOptional.isPresent()) {
            Admin admin = adminOptional.get();
            // 비밀번호 확인
            if (!passwordEncoder.matches(requestDto.password(), admin.getPassword())) {
                throw new IllegalArgumentException("잘못된 비밀번호입니다.");
            }
            // 마지막 로그인 시간 업데이트 및 토큰 생성
            admin.updateLastLogin();
            String token = jwtTokenProvider.createToken(admin.getAdminId());
            // 관리자용 응답 DTO 반환
            return new AdminLoginResponseDto(token, admin.getId(), admin.getAdminId(), admin.getAdminName(), admin.getRole());
        }

        // 3. 어디에도 아이디가 없는 경우
        throw new IllegalArgumentException("가입되지 않은 아이디입니다.");
    }
}