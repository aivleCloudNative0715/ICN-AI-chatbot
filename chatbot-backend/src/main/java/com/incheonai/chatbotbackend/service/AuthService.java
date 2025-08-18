package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.config.jwt.JwtTokenProvider;
import com.incheonai.chatbotbackend.domain.jpa.Admin;
import com.incheonai.chatbotbackend.domain.jpa.User;
import com.incheonai.chatbotbackend.domain.mongodb.ChatSession;
import com.incheonai.chatbotbackend.dto.AdminLoginResponseDto;
import com.incheonai.chatbotbackend.dto.LoginRequestDto;
import com.incheonai.chatbotbackend.dto.LoginResponseDto;
import com.incheonai.chatbotbackend.exception.BusinessException;
import com.incheonai.chatbotbackend.repository.jpa.AdminRepository;
import com.incheonai.chatbotbackend.repository.jpa.UserRepository;
import com.incheonai.chatbotbackend.repository.primary.ChatMessageRepository;
import com.incheonai.chatbotbackend.repository.primary.ChatSessionRepository;
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
import java.util.UUID;
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
    private final ChatSessionRepository chatSessionRepository;
    private final ChatMessageRepository chatMessageRepository;

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
        // Admin 테이블에서 아이디 확인 (관리자는 세션 마이그레이션 없음)
        Optional<Admin> adminOptional = adminRepository.findByAdminId(requestDto.userId());
        if (adminOptional.isPresent()) {
            Admin admin = adminOptional.get();
            if (!passwordEncoder.matches(requestDto.password(), admin.getPassword())) {
                throw new BusinessException(HttpStatus.UNAUTHORIZED, "잘못된 비밀번호입니다.");
            }
            admin.updateLastLogin();
            String token = jwtTokenProvider.createToken(admin.getAdminId());

            redisTemplate.opsForValue().set(token, admin.getAdminId(), jwtTokenProvider.getTokenValidTime(), TimeUnit.MILLISECONDS);
            log.info("관리자 로그인 성공. Redis에 토큰 저장. Key: {}", token);
            return new AdminLoginResponseDto(token, admin.getId(), admin.getAdminId(), admin.getAdminName(), admin.getRole());
        }

        // User 테이블에서 아이디 확인
        User user = userRepository.findByUserId(requestDto.userId())
                .orElseThrow(() -> new BusinessException(HttpStatus.NOT_FOUND, "아이디 또는 비밀번호가 일치하지 않습니다."));

        if (!passwordEncoder.matches(requestDto.password(), user.getPassword())) {
            throw new BusinessException(HttpStatus.UNAUTHORIZED, "잘못된 비밀번호입니다.");
        }
        user.updateLastLogin();

        // ✨ 새로 만든 세션 관리 메서드를 호출하여 정확한 세션 ID를 가져옵니다.
        String sessionId = findOrCreateActiveSessionForUser(user, requestDto.anonymousSessionId());

        String token = jwtTokenProvider.createToken(user.getUserId());
        redisTemplate.opsForValue().set(token, user.getUserId(), jwtTokenProvider.getTokenValidTime(), TimeUnit.MILLISECONDS);
        log.info("로컬 로그인 성공. Redis에 토큰 저장. Key: {}", token);

        return new LoginResponseDto(token, user.getId(), user.getUserId(), user.getGoogleId(), user.getLoginProvider(), sessionId);
    }

    /**
     * ✨ 로그인/회원가입 시 사용자의 최종 세션 ID를 결정하는 핵심 메서드
     * @param user 로그인/가입한 사용자 객체
     * @param anonymousSessionId 프론트에서 전달받은 비회원 세션 ID (null일 수 있음)
     * @return 최종적으로 사용될 세션 ID
     */
    public String findOrCreateActiveSessionForUser(User user, String anonymousSessionId) {
        // 1순위: 비회원 세션 ID가 유효하고, 만료되지 않았으며, 채팅 기록이 존재하면 마이그레이션
        if (anonymousSessionId != null && !anonymousSessionId.isEmpty()) {
            Optional<ChatSession> anonymousSessionOpt = chatSessionRepository.findById(anonymousSessionId);

            if (anonymousSessionOpt.isPresent()) {
                ChatSession session = anonymousSessionOpt.get();

                boolean isSessionActive = session.getExpiresAt() != null && session.getExpiresAt().isAfter(LocalDateTime.now());

                if (isSessionActive && chatMessageRepository.existsBySessionId(anonymousSessionId)) {
                    session.setUserId(String.valueOf(user.getId()));
                    chatSessionRepository.save(session);
                    log.info("채팅 기록이 있는 활성 익명 세션 {}을 사용자 {}에게 마이그레이션했습니다.", session.getId(), user.getUserId());
                    return session.getId();
                }
            }
        }

        // 2순위 & 3순위: (마이그레이션 조건 미충족 시) 사용자의 기존 활성 세션을 찾거나, 없으면 새로 생성
        return findActiveSessionOrCreateNew(user);
    }

    /**
     * ✨ 사용자의 활성 세션을 찾고, 없으면 새로 생성하는 헬퍼 메서드
     */
    private String findActiveSessionOrCreateNew(User user) {
        // 2순위: 사용자의 마지막 활성 세션(24시간 이내)을 찾음
        Optional<ChatSession> existingSession = chatSessionRepository
                .findFirstByUserIdAndExpiresAtAfterOrderByCreatedAtDesc(String.valueOf(user.getId()), LocalDateTime.now());

        if (existingSession.isPresent()) {
            log.info("사용자 {}의 기존 활성 세션 {}을(를) 재사용합니다.", user.getUserId(), existingSession.get().getId());
            return existingSession.get().getId();
        }

        // 3순위: 활성 세션이 없으면 새로 생성
        ChatSession newSession = ChatSession.builder()
                .id(UUID.randomUUID().toString())
                .userId(String.valueOf(user.getId()))
                .createdAt(LocalDateTime.now())
                .lastActivatedAt(LocalDateTime.now())
                .expiresAt(LocalDateTime.now().plusHours(24))
                .build();
        chatSessionRepository.save(newSession);
        log.info("사용자 {}의 새 채팅 세션 {}을(를) 생성합니다.", user.getUserId(), newSession.getId());
        return newSession.getId();
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