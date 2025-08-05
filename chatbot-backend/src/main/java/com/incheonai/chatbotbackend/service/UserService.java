package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.config.jwt.JwtTokenProvider;
import com.incheonai.chatbotbackend.domain.jpa.LoginProvider;
import com.incheonai.chatbotbackend.domain.jpa.User;
import com.incheonai.chatbotbackend.dto.LoginResponseDto;
import com.incheonai.chatbotbackend.dto.SignUpRequestDto;
import com.incheonai.chatbotbackend.exception.BusinessException;
import com.incheonai.chatbotbackend.repository.jpa.AdminRepository;
import com.incheonai.chatbotbackend.repository.jpa.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
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
        return new LoginResponseDto(token, user.getId(), user.getUserId(), user.getGoogleId(), user.getLoginProvider());
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

        // User 엔티티를 LoginResponseDto로 변환하여 반환
        // 프론트엔드에서 이미 토큰을 가지고 있으므로, 여기서는 null로 설정
        return new LoginResponseDto(null, user.getId(), user.getUserId(), user.getGoogleId(), user.getLoginProvider());
    }
}
