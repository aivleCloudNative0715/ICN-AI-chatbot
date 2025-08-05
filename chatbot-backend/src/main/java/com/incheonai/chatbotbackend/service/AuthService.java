package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.config.jwt.JwtTokenProvider;
import com.incheonai.chatbotbackend.domain.jpa.Admin;
import com.incheonai.chatbotbackend.domain.jpa.User;
import com.incheonai.chatbotbackend.dto.AdminLoginResponseDto;
import com.incheonai.chatbotbackend.dto.LoginRequestDto;
import com.incheonai.chatbotbackend.dto.LoginResponseDto;
import com.incheonai.chatbotbackend.exception.BusinessException;
import com.incheonai.chatbotbackend.repository.jpa.AdminRepository;
import com.incheonai.chatbotbackend.repository.jpa.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.Optional;
import java.util.concurrent.TimeUnit;

@Slf4j
@Service
@RequiredArgsConstructor
public class AuthService {

    private final UserRepository userRepository;
    private final AdminRepository adminRepository;
    private final RedisTemplate<String, Object> redisTemplate;
    private final PasswordEncoder passwordEncoder;
    private final JwtTokenProvider jwtTokenProvider;

    /**
     * 아이디 중복을 확인합니다. (재가입 정책, 관리자 계정, Redis 임시 ID 확인 포함)
     */
    public boolean checkIdDuplication(String userId) {
        // 1. Redis에 다른 사용자가 확인 중인 임시 아이디인지 확인
        String redisKey = "temp:userId:" + userId;
        if (Boolean.TRUE.equals(redisTemplate.hasKey(redisKey))) {
            throw new BusinessException(HttpStatus.CONFLICT, "다른 사용자가 확인 중인 아이디입니다.");
        }

        // 2. Admin 테이블에 활성 관리자 계정이 있는지 확인
        if (adminRepository.findByAdminId(userId).isPresent()) {
            return false; // 중복된 아이디가 존재함
        }

        // 3. User 테이블에서 탈퇴한 회원을 포함하여 가장 최근 기록 조회
        Optional<User> lastUserOptional = userRepository.findLastByUserIdWithDeleted(userId);

        if (lastUserOptional.isPresent()) {
            User lastUser = lastUserOptional.get();
            // 3-1. 활성 계정이면 사용 불가
            if (lastUser.getDeletedAt() == null) {
                return false;
            }
            // 3-2. 탈퇴한 계정이면 30일 경과 여부 확인
            long daysSinceDeletion = ChronoUnit.DAYS.between(lastUser.getDeletedAt(), LocalDateTime.now());
            if (daysSinceDeletion < 30) {
                long daysLeft = 30 - daysSinceDeletion;
                throw new BusinessException(HttpStatus.CONFLICT, "탈퇴 후 30일이 지나지 않은 아이디입니다. (" + daysLeft + "일 후 사용 가능)");
            }
        }

        // 4. 모든 중복 검사를 통과했으면 1분간 임시 아이디로 Redis에 저장
        redisTemplate.opsForValue().set(redisKey, "true", 1, TimeUnit.MINUTES);

        // 최종적으로 사용 가능한 아이디임
        return true;
    }

    @Transactional
    public Object login(LoginRequestDto requestDto) {
        // 1. User 테이블에서 아이디 확인
        Optional<User> userOptional = userRepository.findByUserId(requestDto.userId());
        if (userOptional.isPresent()) {
            User user = userOptional.get();
            // 비밀번호 확인
            if (!passwordEncoder.matches(requestDto.password(), user.getPassword())) {
                throw new BusinessException(HttpStatus.UNAUTHORIZED, "잘못된 비밀번호입니다.");
            }
            // 마지막 로그인 시간 업데이트 및 토큰 생성
            user.updateLastLogin();
            String token = jwtTokenProvider.createToken(user.getUserId());

            redisTemplate.opsForValue().set(token, user.getUserId(), jwtTokenProvider.getTokenValidTime(), TimeUnit.MILLISECONDS);
            log.info("로컬 로그인 성공. Redis에 토큰 저장. Key: {}", token);

            // 사용자용 응답 DTO 반환
            return new LoginResponseDto(token, user.getId(), user.getUserId(), user.getGoogleId(), user.getLoginProvider());
        }

        // 2. Admin 테이블에서 아이디 확인
        Optional<Admin> adminOptional = adminRepository.findByAdminId(requestDto.userId());
        if (adminOptional.isPresent()) {
            Admin admin = adminOptional.get();
            // 비밀번호 확인
            if (!passwordEncoder.matches(requestDto.password(), admin.getPassword())) {
                throw new BusinessException(HttpStatus.UNAUTHORIZED, "잘못된 비밀번호입니다.");
            }
            // 마지막 로그인 시간 업데이트 및 토큰 생성
            admin.updateLastLogin();
            String token = jwtTokenProvider.createToken(admin.getAdminId());

            redisTemplate.opsForValue().set(token, admin.getAdminId(), jwtTokenProvider.getTokenValidTime(), TimeUnit.MILLISECONDS);
            log.info("관리자 로그인 성공. Redis에 토큰 저장. Key: {}", token);
            // 관리자용 응답 DTO 반환
            return new AdminLoginResponseDto(token, admin.getId(), admin.getAdminId(), admin.getAdminName(), admin.getRole());
        }

        // 3. 어디에도 아이디가 없는 경우
        throw new BusinessException(HttpStatus.NOT_FOUND, "아이디 또는 비밀번호가 일치하지 않습니다.");
    }

    public void logout(String accessToken) {
        // 1. 토큰 유효성 검증
        if (!jwtTokenProvider.validateToken(accessToken)) {
            throw new BusinessException(HttpStatus.UNAUTHORIZED, "유효하지 않은 토큰입니다.");
        }

        // 2. Redis에 로그아웃된 토큰으로 저장
        if (Boolean.TRUE.equals(redisTemplate.hasKey(accessToken))) {
            redisTemplate.delete(accessToken);
            log.info("로그아웃 처리 완료. Redis에서 토큰 삭제. Key: {}", accessToken);
        } else {
            log.warn("로그아웃 요청된 토큰이 Redis에 존재하지 않습니다. Key: {}", accessToken);
        }
    }
}