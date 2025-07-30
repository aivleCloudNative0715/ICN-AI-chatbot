package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.config.jwt.JwtTokenProvider;
import com.incheonai.chatbotbackend.domain.jpa.LoginProvider;
import com.incheonai.chatbotbackend.domain.jpa.User;
import com.incheonai.chatbotbackend.dto.LoginRequestDto;
import com.incheonai.chatbotbackend.dto.LoginResponseDto;
import com.incheonai.chatbotbackend.dto.SignUpRequestDto;
import com.incheonai.chatbotbackend.repository.jpa.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor // final 필드에 대한 생성자를 자동으로 생성
public class UserService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtTokenProvider jwtTokenProvider;

    @Transactional
    public LoginResponseDto signup(SignUpRequestDto requestDto) {
        // 1. 아이디 중복 확인
        if (userRepository.findByUserId(requestDto.userId()).isPresent()) {
            throw new IllegalArgumentException("이미 사용 중인 아이디입니다.");
        }

        // 2. 비밀번호 암호화
        String encodedPassword = passwordEncoder.encode(requestDto.password());

        // 3. 사용자 정보 생성
        User user = User.builder()
                .userId(requestDto.userId())
                .password(encodedPassword)
                .loginProvider(LoginProvider.LOCAL)
                .build();

        // 4. 사용자 정보 저장
        userRepository.save(user);

        // 5. JWT 토큰 생성 및 반환 (자동 로그인)
        String token = jwtTokenProvider.createToken(user.getUserId());
        return new LoginResponseDto(token, null, null, null, null);
    }

    @Transactional
    public LoginResponseDto login(LoginRequestDto requestDto) {
        User user = userRepository.findByUserId(requestDto.userId())
                .orElseThrow(() -> new IllegalArgumentException("가입되지 않은 아이디입니다."));

        if (!passwordEncoder.matches(requestDto.password(), user.getPassword())) {
            throw new IllegalArgumentException("잘못된 비밀번호입니다.");
        }

        user.updateLastLogin();

        String token = jwtTokenProvider.createToken(user.getUserId());
        return new LoginResponseDto(token, null, null, null, null);
    }
}
