package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.config.jwt.JwtTokenProvider;
import com.incheonai.chatbotbackend.domain.jpa.LoginProvider;
import com.incheonai.chatbotbackend.domain.jpa.User;
import com.incheonai.chatbotbackend.domain.mongodb.ChatSession;
import com.incheonai.chatbotbackend.dto.LoginResponseDto;
import com.incheonai.chatbotbackend.dto.SignUpRequestDto;
import com.incheonai.chatbotbackend.exception.BusinessException;
import com.incheonai.chatbotbackend.repository.jpa.AdminRepository;
import com.incheonai.chatbotbackend.repository.jpa.UserRepository;
import com.incheonai.chatbotbackend.repository.primary.ChatSessionRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.concurrent.TimeUnit;

@Slf4j
@Service
@RequiredArgsConstructor
public class UserService {

    private final UserRepository userRepository;
    private final AdminRepository adminRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtTokenProvider jwtTokenProvider;
    private final RedisTemplate<String, Object> redisTemplate;
    private final ChatSessionRepository chatSessionRepository;
    private final AuthService authService;

    @Transactional
    public LoginResponseDto signup(SignUpRequestDto requestDto) {
        // 비밀번호, 비밀번호 재확인 일치 확인
        if (!requestDto.password().equals(requestDto.passwordConfirm())) {
            throw new BusinessException(HttpStatus.BAD_REQUEST, "비밀번호가 일치하지 않습니다.");
        }

        // 1. 최종적으로 아이디 중복 확인 (DB 동시성 문제 방지)
        if (userRepository.findByUserId(requestDto.userId()).isPresent() || adminRepository.findByAdminId(requestDto.userId()).isPresent()) {
            throw new BusinessException(HttpStatus.CONFLICT, "이미 사용 중인 아이디입니다.");
        }

        // 2. 비밀번호 암호화
        String encodedPassword = passwordEncoder.encode(requestDto.password());

        // 3. 사용자 정보 생성 및 저장
        User user = User.builder()
                .userId(requestDto.userId())
                .password(encodedPassword)
                .loginProvider(LoginProvider.LOCAL)
                .build();
        userRepository.save(user);

        // 4. Redis에 저장된 임시 아이디 삭제
        String redisKey = "temp:userId:" + requestDto.userId();
        if (Boolean.TRUE.equals(redisTemplate.hasKey(redisKey))) {
            redisTemplate.delete(redisKey);
        }

        // 5. JWT 토큰 생성 및 반환 (자동 로그인)
        String token = jwtTokenProvider.createToken(user.getUserId());

        redisTemplate.opsForValue().set(
                token,
                user.getUserId(),
                jwtTokenProvider.getTokenValidTime(),
                TimeUnit.MILLISECONDS
        );
        log.info("회원가입 성공. 자동 로그인을 위해 Redis에 토큰 저장. Key: {}", token);

        // 6. 회원가입과 동시에 새로운 채팅 세션 생성
        String sessionId = authService.findOrCreateActiveSessionForUser(user, requestDto.anonymousSessionId());

        return new LoginResponseDto(token, user.getId(), user.getUserId(), user.getGoogleId(), user.getLoginProvider(), sessionId);
    }

    /**
     * 사용자 정보 조회 메서드
     * @param userId (JWT 토큰에서 추출한 사용자 ID)
     * @return LoginResponseDto (토큰은 제외하고 사용자 정보만 담음)
     */
    @Transactional(readOnly = true)
    public LoginResponseDto getUserInfo(String userId) {
        User user = userRepository.findByUserId(userId)
                .orElseThrow(() -> new BusinessException(HttpStatus.NOT_FOUND, "사용자 정보를 찾을 수 없습니다."));

        // 사용자의 활성 세션 조회
        String activeSessionId = chatSessionRepository
                .findFirstByUserIdAndExpiresAtAfterOrderByCreatedAtDesc(String.valueOf(user.getId()), LocalDateTime.now())
                .map(ChatSession::getId) // 세션이 존재하면 ID를 꺼내고
                .orElse(null); // 없으면 null

        // User 엔티티를 LoginResponseDto로 변환하여 반환
        // 프론트엔드에서 이미 토큰을 가지고 있으므로, 여기서는 null로 설정
        return new LoginResponseDto(null, user.getId(), user.getUserId(), user.getGoogleId(), user.getLoginProvider(), activeSessionId);
    }

    /**
     * 현재 로그인된 사용자의 계정을 삭제(soft-delete)합니다.
     * @param userId (JWT 토큰에서 추출한 사용자 ID)
     */
    @Transactional
    public void deleteAccount(String userId) {
        // 1. 사용자 ID로 사용자를 찾습니다.
        User user = userRepository.findByUserId(userId)
                .orElseThrow(() -> new BusinessException(HttpStatus.NOT_FOUND, "계정 삭제 실패: 사용자 정보를 찾을 수 없습니다."));

        // 2. JpaRepository의 delete를 호출합니다.
        // User 엔티티의 @SQLDelete 어노테이션 덕분에 실제로는 UPDATE 쿼리가 실행됩니다.
        userRepository.delete(user);
    }
}
