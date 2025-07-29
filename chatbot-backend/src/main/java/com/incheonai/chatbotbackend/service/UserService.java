package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.domain.LoginProvider;
import com.incheonai.chatbotbackend.domain.User;
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

    @Transactional
    public void signup(SignUpRequestDto requestDto) {
        // 1. 아이디 중복 확인
        if (userRepository.findByUserId(requestDto.userId()).isPresent()) {
            throw new IllegalArgumentException("이미 사용 중인 아이디입니다.");
        }

        // 2. 비밀번호 암호화
        String encodedPassword = passwordEncoder.encode(requestDto.password());

        // 3. 사용자 정보 생성 및 저장
        User user = User.builder()
                .userId(requestDto.userId())
                .password(encodedPassword)
                .loginProvider(LoginProvider.LOCAL) // 로컬 회원가입
                .build();

        userRepository.save(user);
    }
}
