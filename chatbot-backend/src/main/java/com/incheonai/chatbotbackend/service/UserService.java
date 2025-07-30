package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.config.jwt.JwtTokenProvider;
import com.incheonai.chatbotbackend.domain.jpa.LoginProvider;
import com.incheonai.chatbotbackend.domain.jpa.User;
import com.incheonai.chatbotbackend.dto.LoginResponseDto;
import com.incheonai.chatbotbackend.dto.SignUpRequestDto;
import com.incheonai.chatbotbackend.repository.jpa.AdminRepository;
import com.incheonai.chatbotbackend.repository.jpa.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class UserService {

    private final UserRepository userRepository;
    private final AdminRepository adminRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtTokenProvider jwtTokenProvider;
    private final RedisTemplate<String, Object> redisTemplate;

    @Transactional
    public LoginResponseDto signup(SignUpRequestDto requestDto) {
        // 1. 최종적으로 아이디 중복 확인 (DB 동시성 문제 방지)
        if (userRepository.findByUserId(requestDto.userId()).isPresent() || adminRepository.findByAdminId(requestDto.userId()).isPresent()) {
            throw new IllegalArgumentException("이미 사용 중인 아이디입니다.");
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
        return new LoginResponseDto(token, user.getId(), user.getUserId(), user.getGoogleId(), user.getLoginProvider());
    }
}
